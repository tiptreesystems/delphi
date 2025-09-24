import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from collections import defaultdict


@dataclass
class Question:
    id: str
    source: str
    question: str
    resolution_criteria: str
    background: str
    market_info_open_datetime: str
    market_info_close_datetime: str
    market_info_resolution_criteria: str
    url: str
    freeze_datetime: str
    freeze_datetime_value: Union[str, float]
    freeze_datetime_value_explanation: str
    source_intro: str
    combination_of: str
    resolution_dates: str
    topic: Optional[str] = None


@dataclass
class Resolution:
    id: Union[str, List[str]]
    source: str
    direction: Optional[str]
    resolution_date: str
    resolved_to: float
    resolved: bool


@dataclass
class Forecast:
    id: str
    source: str
    forecast: float
    resolution_date: Optional[str]
    reasoning: str
    direction: Optional[str]
    user_id: str
    searches: Optional[List[str]] = None
    consulted_urls: Optional[List[str]] = None


class ForecastDataLoader:
    def __init__(self, data_dir: str = "dataset/datasets"):
        self.data_dir = Path(data_dir)
        self.questions: Dict[str, Question] = {}
        self.resolutions: Dict[str, Resolution] = {}
        self.super_forecasts: Dict[str, List[Forecast]] = {}
        self.public_forecasts: Dict[str, List[Forecast]] = {}

        self._load_questions()
        self._load_resolutions()
        self._load_super_forecasts()
        self._load_public_forecasts()

    def _load_questions(self):
        question_file = self.data_dir / "question_sets" / "2024-07-21-human.json"
        nomic_csv_path = (
            self.data_dir.parent
            / "nomic"
            / "Prediction Market Questions_250721-213003.csv"
        )  # TODO: convert to json
        if question_file.exists():
            with open(question_file, "r") as f:
                data = json.load(f)
                for q_data in data["questions"]:
                    question = Question(**q_data)
                    self._add_topic(question)
                    self.questions[question.id] = question

    def _add_topic(self, question: Question):
        if not hasattr(self, "_nomic_df"):
            nomic_csv_path = (
                self.data_dir.parent
                / "nomic"
                / "Prediction Market Questions_250721-213003.csv"
            )
            if nomic_csv_path.exists():
                self._nomic_df = pd.read_csv(nomic_csv_path)
            else:
                self._nomic_df = None

        if self._nomic_df is not None:
            row = self._nomic_df[self._nomic_df["id"] == question.id]
            if not row.empty and "Nomic Topic: broad" in row.columns:
                question.topic = row.iloc[0]["Nomic Topic: broad"]

    def _load_resolutions(self):
        resolution_file = (
            self.data_dir / "resolution_sets" / "2024-07-21_resolution_set.json"
        )
        grouped = defaultdict(list)

        if resolution_file.exists():
            with open(resolution_file, "r") as f:
                data = json.load(f)

            for r in data.get("resolutions", []):
                ids = r["id"] if isinstance(r["id"], list) else [r["id"]]
                directions = r.get("direction")

                for i, qid in enumerate(ids):
                    # make a per-question entry
                    r_item = dict(r)
                    r_item["id"] = qid
                    if isinstance(directions, list):
                        r_item["direction"] = directions[i]
                    resolution = Resolution(**r_item)
                    grouped[qid].append(resolution)

            # sort each list by resolution_date (ascending)
            for qid in grouped:
                grouped[qid].sort(key=lambda res: res.resolution_date)

        self.resolutions = grouped
        return grouped

    def _load_super_forecasts(self):
        forecast_file = (
            self.data_dir
            / "forecast_sets"
            / "2024-07-21"
            / "2024-07-21.ForecastBench.human_super_individual.json"
        )
        if forecast_file.exists():
            with open(forecast_file, "r") as f:
                data = json.load(f)
                for f_data in data["forecasts"]:
                    forecast = Forecast(**f_data)
                    if forecast.id not in self.super_forecasts:
                        self.super_forecasts[forecast.id] = []
                    self.super_forecasts[forecast.id].append(forecast)

    def _load_public_forecasts(self):
        public_forecast_file = (
            self.data_dir
            / "forecast_sets"
            / "2024-07-21"
            / "2024-07-21.ForecastBench.human_public_individual.json"
        )
        if public_forecast_file.exists():
            with open(public_forecast_file, "r") as f:
                data = json.load(f)
                for f_data in data["forecasts"]:
                    forecast = Forecast(**f_data)
                    if forecast.id not in self.public_forecasts:
                        self.public_forecasts[forecast.id] = []
                    self.public_forecasts[forecast.id].append(forecast)

    def get_question(self, question_id: str) -> Optional[Question]:
        return self.questions.get(question_id)

    def get_all_questions(self) -> List[Question]:
        return list(self.questions.values())

    def get_same_topic_questions(self, question_id: str) -> List[Question]:
        question = self.get_question(question_id)
        if not question or not question.topic:
            return []
        return [
            q
            for q in self.questions.values()
            if q.id != question_id and q.topic == question.topic
        ]

    def search_questions(self, keyword: str) -> List[Question]:
        keyword_lower = keyword.lower()
        results = []
        for question in self.questions.values():
            if (
                keyword_lower in question.question.lower()
                or keyword_lower in question.background.lower()
                or keyword_lower in question.resolution_criteria.lower()
            ):
                results.append(question)
        return results

    def is_resolved(self, question_id: str, resolution_date: str) -> bool:
        """Return True if the given question_id has a resolution on the given date and it is resolved."""
        resolutions = self.resolutions.get(question_id)
        if not resolutions:
            return False
        for res in resolutions:
            if str(res.resolution_date) == resolution_date:
                return bool(res.resolved)
        return False

    def get_resolution(
        self, question_id: str, resolution_date: str
    ) -> Optional[Resolution]:
        """Return the Resolution for a given question_id and resolution_date (string)."""
        resolutions = self.resolutions.get(question_id)
        if not resolutions:
            return None
        for res in resolutions:
            if str(res.resolution_date) == resolution_date:
                return res
        return None

    def get_super_forecasts(
        self,
        *,  # make sure to use keyword arguments
        question_id: Optional[str] = None,
        resolution_date: Optional[str] = None,
        user_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Forecast]:
        if question_id is not None:
            forecasts = self.super_forecasts.get(question_id, [])
        else:
            # Flatten all forecasts if question_id is not provided
            forecasts = [f for flist in self.super_forecasts.values() for f in flist]
        if resolution_date is not None:
            forecasts = [f for f in forecasts if f.resolution_date == resolution_date]
        if user_id is not None:
            forecasts = [f for f in forecasts if f.user_id == user_id]
        if topic is not None:
            # Filter forecasts by topic using the question's topic
            forecasts = [
                f
                for f in forecasts
                if (q := self.questions.get(f.id)) is not None and q.topic == topic
            ]
        return forecasts

    def get_public_forecasts(
        self,
        *,  # make sure to use keyword arguments
        question_id: Optional[str] = None,
        resolution_date: Optional[str] = None,
        user_id: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> List[Forecast]:
        if question_id is not None:
            forecasts = self.public_forecasts.get(question_id, [])
        else:
            # Flatten all forecasts if question_id is not provided
            forecasts = [f for flist in self.public_forecasts.values() for f in flist]
        if resolution_date is not None:
            forecasts = [f for f in forecasts if f.resolution_date == resolution_date]
        if user_id is not None:
            forecasts = [f for f in forecasts if f.user_id == user_id]
        if topic is not None:
            # Filter forecasts by topic using the question's topic
            forecasts = [
                f
                for f in forecasts
                if (q := self.questions.get(f.id)) is not None and q.topic == topic
            ]
        return forecasts

    # def get_question_with_forecasts(self, question_id: str) -> Optional[Dict]:
    #     question = self.get_question(question_id)
    #     if not question:
    #         print("Question not found:", question_id)

    #     return {
    #         'question': question,
    #         'resolution': self.get_resolution(question_id),
    #         'super_forecasts': self.get_super_forecasts(question_id=question_id),
    #         'public_forecasts': self.get_public_forecasts(question_id),
    #         'is_resolved': self.is_resolved(question_id)
    #     }

    def sample_random_question(self) -> Optional[Question]:
        import random

        if self.questions:
            return random.choice(list(self.questions.values()))
        return None

    def get_resolved_questions(self) -> List[Question]:
        resolved_ids = {
            res_id for res_id, res in self.resolutions.items() if res.resolved
        }
        return [q for q_id, q in self.questions.items() if q_id in resolved_ids]

    def get_unresolved_questions(self) -> List[Question]:
        resolved_ids = {
            res_id for res_id, res in self.resolutions.items() if res.resolved
        }
        return [q for q_id, q in self.questions.items() if q_id not in resolved_ids]

    def get_questions_with_topics(self) -> List[Question]:
        return [
            q
            for q in self.questions.values()
            if q.topic is not None and isinstance(q.resolution_dates, list)
        ]  # TODO: improve resolution date handling

    def get_statistics(self) -> Dict:
        total_questions = len(self.questions)
        resolved_count = len(self.get_resolved_questions())
        unresolved_count = len(self.get_unresolved_questions())

        total_forecasts = sum(
            len(forecasts) for forecasts in self.super_forecasts.values()
        )
        questions_with_forecasts = len(self.super_forecasts)

        return {
            "total_questions": total_questions,
            "resolved_questions": resolved_count,
            "unresolved_questions": unresolved_count,
            "total_forecasts": total_forecasts,
            "questions_with_forecasts": questions_with_forecasts,
            "average_forecasts_per_question": total_forecasts / questions_with_forecasts
            if questions_with_forecasts > 0
            else 0,
        }

    def get_search_and_url_statistics(self) -> Dict:
        total_forecasts = 0
        forecasts_with_searches = 0
        forecasts_with_urls = 0
        total_searches = 0
        total_urls = 0
        search_counts = []
        url_counts = []

        for forecast_list in self.super_forecasts.values():
            for forecast in forecast_list:
                total_forecasts += 1

                if forecast.searches:
                    forecasts_with_searches += 1
                    search_count = len(forecast.searches)
                    total_searches += search_count
                    search_counts.append(search_count)

                if forecast.consulted_urls:
                    forecasts_with_urls += 1
                    url_count = len(forecast.consulted_urls)
                    total_urls += url_count
                    url_counts.append(url_count)

        return {
            "total_forecasts": total_forecasts,
            "forecasts_with_searches": forecasts_with_searches,
            "forecasts_with_urls": forecasts_with_urls,
            "percentage_with_searches": (
                forecasts_with_searches / total_forecasts * 100
            )
            if total_forecasts > 0
            else 0,
            "percentage_with_urls": (forecasts_with_urls / total_forecasts * 100)
            if total_forecasts > 0
            else 0,
            "total_searches": total_searches,
            "total_urls": total_urls,
            "avg_searches_when_present": (total_searches / forecasts_with_searches)
            if forecasts_with_searches > 0
            else 0,
            "avg_urls_when_present": (total_urls / forecasts_with_urls)
            if forecasts_with_urls > 0
            else 0,
            "min_searches": min(search_counts) if search_counts else 0,
            "max_searches": max(search_counts) if search_counts else 0,
            "min_urls": min(url_counts) if url_counts else 0,
            "max_urls": max(url_counts) if url_counts else 0,
        }

    def get_all_topics(self) -> List[str]:
        topics = set()
        for question in self.questions.values():
            if question.topic:
                topics.add(question.topic)
        return sorted(topics)

    def get_topic_related_users(self, topic: str) -> Dict[str, int]:
        user_forecast_count = {}
        for question in self.questions.values():
            if question.topic == topic:
                for forecast in self.super_forecasts.get(question.id, []):
                    if forecast.user_id not in user_forecast_count:
                        user_forecast_count[forecast.user_id] = 0
                    user_forecast_count[forecast.user_id] += 1

    def get_topic(self, question_id: str) -> Optional[str]:
        question = self.get_question(question_id)
        if question and question.topic:
            return question.topic
        return None


if __name__ == "__main__":
    loader = ForecastDataLoader()

    print("Dataset Statistics:")
    stats = loader.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nSearch and URL Statistics:")
    search_stats = loader.get_search_and_url_statistics()
    for key, value in search_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nSample Question:")
    sample = loader.sample_random_question()
    if sample:
        print(f"  ID: {sample.id}")
        print(f"  Question: {sample.question}")
        print(f"  Resolved: {loader.is_resolved(sample.id)}")

        super_forecasts = loader.get_super_forecasts(question_id=sample.id)
        public_forecasts = loader.get_public_forecasts(question_id=sample.id)
        if super_forecasts:
            print(f"  Number of super forecasts: {len(super_forecasts)}")
            print(
                f"  Average super forecast: {sum(f.forecast for f in super_forecasts) / len(super_forecasts):.3f}"
            )
        if public_forecasts:
            print(f"  Number of public forecasts: {len(public_forecasts)}")
            print(
                f"  Average public forecast: {sum(f.forecast for f in public_forecasts) / len(public_forecasts):.3f}"
            )
