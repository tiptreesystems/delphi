import json
import sys
import types
import unittest
from datetime import datetime, timedelta
from typing import Any, Dict, List

from unittest.mock import patch


import debugpy

print("Waiting for debugger attach on port 5679...")
debugpy.listen(5679)
debugpy.wait_for_client()
print("Debugger attached, running tests...")

from agents import btf_tools as btf_tools_module
from agents.btf_agent import BTFAgent
from agents.btf_utils import BTFQuestion, BTFSearchFact, BTFSearchResult
from utils.models import BaseLLM


class FakeFunctionBlock:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, name: str, arguments: Dict[str, Any], call_id: str = "call_1"):
        self.id = call_id
        self.type = "function"
        self.function = FakeFunctionBlock(name, json.dumps(arguments))


class FakeMessage:
    def __init__(
        self,
        content: str | None = None,
        tool_calls: List[FakeToolCall] | None = None,
    ) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []


class FakeChoice:
    def __init__(self, message: FakeMessage) -> None:
        self.message = message


class FakeResponse:
    def __init__(self, message: FakeMessage) -> None:
        self.choices = [FakeChoice(message)]


class FakeChatCompletions:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self._responses = responses
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs):
        call_index = len(self.calls)
        if call_index >= len(self._responses):
            raise RuntimeError("No more responses configured for FakeChatCompletions.")
        self.calls.append(kwargs)
        return self._responses[call_index]


class FakeChat:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.completions = FakeChatCompletions(responses)


class FakeOpenAIClient:
    def __init__(self, responses: List[FakeResponse]) -> None:
        self.chat = FakeChat(responses)


class FakeLLM(BaseLLM):
    def __init__(self, responses: List[FakeResponse]):
        super().__init__(api_key=None, system_prompt="You are a test assistant.")
        self.client = FakeOpenAIClient(responses)
        self.model = "gpt-4o"

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - not used
        raise NotImplementedError

    def generate_stream(self, prompt: str, **kwargs):  # pragma: no cover - not used
        raise NotImplementedError


class BTFAgentTestCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        now = datetime.utcnow()
        self.question = BTFQuestion(
            id="Q1",
            question="Will a major climate agreement be signed this year?",
            background="Leaders are negotiating international accords.",
            resolution_criteria="Agreement must be signed by major economies.",
            scoring_weight=1.0,
            fine_print="Includes provisional agreements.",
            resolution="unknown",
            resolved_at="",
            present_date=now,
            date_cutoff_start=now - timedelta(days=30),
            date_cutoff_end=now,
        )

        async def fake_generate_search_queries(btf_question: BTFQuestion):
            self.assertEqual(btf_question.present_date, self.question.present_date)
            self.assertEqual(
                btf_question.date_cutoff_end, self.question.date_cutoff_end
            )
            return ["climate agreement 2024 outlook"]

        async def fake_get_retrosearch_results(
            query: str,
            date_cutoff_start: datetime,
            date_cutoff_end: datetime,
            max_results: int = 5,
        ):
            self.assertEqual(date_cutoff_start, self.question.date_cutoff_start)
            self.assertEqual(date_cutoff_end, self.question.date_cutoff_end)
            return [
                {
                    "title": "Negotiations Intensify",
                    "link": "https://example.com/article",
                    "snippet": "Diplomats see momentum toward a deal.",
                }
            ]

        async def fake_get_result_content(
            url: str, date_cutoff_start: datetime, date_cutoff_end: datetime
        ):
            self.assertEqual(date_cutoff_start, self.question.date_cutoff_start)
            self.assertEqual(date_cutoff_end, self.question.date_cutoff_end)
            return "Content describing the likelihood of an agreement."

        async def fake_extract_evidence_from_page(
            search_result: BTFSearchResult, *_args, **_kwargs
        ):
            return [
                BTFSearchFact(
                    title=search_result.title,
                    url=search_result.url,
                    fact="Officials indicate a signing ceremony is scheduled.",
                )
            ]

        self._patches = [
            patch.object(
                btf_tools_module,
                "generate_search_queries",
                fake_generate_search_queries,
            ),
            patch.object(
                btf_tools_module,
                "get_retrosearch_results",
                fake_get_retrosearch_results,
            ),
            patch.object(
                btf_tools_module,
                "get_result_content",
                fake_get_result_content,
            ),
            patch.object(
                btf_tools_module,
                "extract_evidence_from_page",
                fake_extract_evidence_from_page,
            ),
        ]

        for _patch in self._patches:
            _patch.start()

    async def asyncTearDown(self) -> None:
        for _patch in self._patches:
            _patch.stop()

    async def test_btf_agent_runs_tool_loop(self):
        tool_call_message = FakeMessage(
            tool_calls=[
                FakeToolCall(
                    "btf_generate_search_queries",
                    {
                        "question_id": self.question.id,
                        "question": self.question.question,
                        "max_queries": 2,
                    },
                )
            ]
        )
        final_message = FakeMessage(
            content="After reviewing the evidence, FINAL PROBABILITY: 0.42"
        )

        fake_llm = FakeLLM(
            [FakeResponse(tool_call_message), FakeResponse(final_message)]
        )
        agent = BTFAgent(fake_llm)

        result = await agent.forecast(self.question)

        self.assertAlmostEqual(result.probability, 0.42, places=3)
        self.assertTrue(result.response.endswith("FINAL PROBABILITY: 0.42"))

        tool_names = [entry["name"] for entry in result.tool_outputs]
        self.assertIn("btf_generate_search_queries", tool_names)

        queries_entry = next(
            entry
            for entry in result.tool_outputs
            if entry["name"] == "btf_generate_search_queries"
        )
        self.assertEqual(
            queries_entry["content"]["queries"],
            ["climate agreement 2024 outlook"],
        )

        recorded_calls = fake_llm.client.chat.completions.calls
        self.assertEqual(len(recorded_calls), 2)
        self.assertTrue(recorded_calls[0].get("tools"))


def load_tests(
    loader: unittest.TestLoader, tests: unittest.TestSuite, pattern: str
) -> unittest.TestSuite:
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(BTFAgentTestCase))
    return suite


if __name__ == "__main__":
    unittest.main()
