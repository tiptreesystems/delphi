from typing import Optional, Dict
from utils.models import BaseLLM, ConversationManager
from dataset.dataloader import Question
import dotenv
from utils.prompt_loader import load_prompt, get_prompt_loader
from utils.models import ClaudeLLM

dotenv.load_dotenv()

# Load prompts
prompt_loader = get_prompt_loader()


class Mediator:
    """
    Minimal Delphi Mediator:
    - holds its own ConversationManager
    - receives & stores expert messages
    - crafts a single feedback memo
    """

    def __init__(self, llm: BaseLLM, config: Optional[dict] = None):
        self.llm = llm
        self.config = config or {}
        self.conversation_manager = ConversationManager(llm)
        # Load system prompt from reference, not inline text
        prompt_name = self.config.get("system_prompt_name", "mediator_system")
        prompt_version = self.config.get("system_prompt_version", "v1")
        print(f"Loading system prompt: {prompt_name} {prompt_version}")
        self.system_prompt = load_prompt(prompt_name, prompt_version)
        self.max_tokens = self.config.get("feedback_max_tokens", 800)
        self.temperature = self.config.get("feedback_temperature", 0.2)

        # Local store of messages (expert_id -> text)
        self.expert_messages: Dict[str, str] = {}

    # ---- message intake ----

    def reset(self) -> None:
        """Clear mediator and conversation state."""
        self.expert_messages.clear()
        self.conversation_manager.messages.clear()

    def receive_message(self, *, expert_id: str, message: Dict[str, str]) -> None:
        """
        Add/replace a single expert's message.
        Accepts a message dict with keys 'role' and 'content'.
        """
        if not isinstance(message, dict) or "content" not in message:
            raise ValueError("Message must be a dict with at least a 'content' key.")
        self.expert_messages[expert_id] = message["content"]

    def receive_messages(self, messages: Dict[str, Dict[str, str]]) -> None:
        """
        Bulk add/replace expert messages.
        Accepts a dict mapping expert_id -> message dict with keys 'role' and 'content'.
        """
        for expert_id, message in messages.items():
            if not isinstance(message, dict) or "content" not in message:
                raise ValueError(f"Message for expert {expert_id} must be a dict with at least a 'content' key.")
            self.expert_messages[expert_id] = message["content"]

    # ---- feedback ----
    def start_round(self, *, round_idx: int, question: Question, extra_context: Optional[str] = None) -> None:
        """Append a new round header + question context without clearing prior messages."""
        # Seed a system message once if none exists yet (but not for Claude, which handles system separately)
        if not isinstance(self.llm, ClaudeLLM) and not any(m["role"] == "system" for m in self.conversation_manager.messages):
            self.conversation_manager.add_message(role="system", content=self.system_prompt)

        # Build optional sections
        url_section = f"\nURL: {question.url}" if getattr(question, "url", None) else ""
        freeze_value_section = f"\nMARKET FREEZE VALUE: {question.freeze_datetime_value}" if getattr(question, "freeze_datetime_value", None) else ""
        freeze_explanation_section = f"\nMARKET FREEZE VALUE EXPLANATION: {question.freeze_datetime_value_explanation}" if getattr(question, "freeze_datetime_value_explanation", None) else ""
        extra_context_section = f"\n\nADDITIONAL CONTEXT:\n{extra_context}" if extra_context else ""
        
        # Load and format the round template
        round_content = load_prompt(
            'delphi_round',
            'v1',
            round_idx=round_idx,
            question=question.question,
            background=question.background,
            resolution_criteria=question.resolution_criteria,
            url_section=url_section,
            freeze_value_section=freeze_value_section,
            freeze_explanation_section=freeze_explanation_section,
            extra_context_section=extra_context_section
        )

        self.conversation_manager.add_message(role="system", content=round_content)


    async def generate_feedback(
        self,
        *,
        reset: bool = False
    ) -> str:
        """
        Append a feedback request based on all accumulated expert messages & prior rounds.
        Set reset=True to start a fresh thread (rare).
        """
        if reset:
            self.conversation_manager.messages.clear()

        # Ensure there is a system message (but not for Claude, which handles system separately)
        if not isinstance(self.llm, ClaudeLLM) and not any(m["role"] == "system" for m in self.conversation_manager.messages):
            self.conversation_manager.add_message(role="system", content=self.system_prompt)

        # Append expert messages for this turn (anonymized, deterministic order)
        if self.expert_messages:
            for eid in sorted(self.expert_messages.keys()):
                message = self.expert_messages[eid]
                content = message['content'] if isinstance(message, dict) else message

                self.conversation_manager.add_message(
                    role="user",
                    content=f"[EXPERT {eid}] {content}"
                )

        # Append the mediator synthesis request (annotate round if provided)
        mediator_request = load_prompt(
            'mediator_feedback',
            'v1',
        )

        feedback = await self.conversation_manager.generate_response(
            mediator_request,
            input_message_type="user",
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )

        return feedback.strip()
