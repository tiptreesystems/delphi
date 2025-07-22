import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime


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
    def __init__(self, data_dir: str = "data"):
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
        question_file = self.data_dir / "2024-07-21-human.json"
        if question_file.exists():
            with open(question_file, 'r') as f:
                data = json.load(f)
                for q_data in data['questions']:
                    question = Question(**q_data)
                    self.questions[question.id] = question
    
    def _load_resolutions(self):
        resolution_file = self.data_dir / "2024-07-21_resolution_set.json"
        if resolution_file.exists():
            with open(resolution_file, 'r') as f:
                data = json.load(f)
                for r_data in data['resolutions']:
                    resolution = Resolution(**r_data)
                    if isinstance(resolution.id, str):
                        existing = self.resolutions.get(resolution.id)
                        if (existing is None or 
                            resolution.resolution_date > existing.resolution_date):
                            self.resolutions[resolution.id] = resolution
                    elif isinstance(resolution.id, list):
                        for res_id in resolution.id:
                            existing = self.resolutions.get(res_id)
                            if (existing is None or 
                                resolution.resolution_date > existing.resolution_date):
                                self.resolutions[res_id] = resolution
    
    def _load_super_forecasts(self):
        forecast_file = self.data_dir / "2024-07-21.ForecastBench.human_super_individual.json"
        if forecast_file.exists():
            with open(forecast_file, 'r') as f:
                data = json.load(f)
                for f_data in data['forecasts']:
                    forecast = Forecast(**f_data)
                    if forecast.id not in self.super_forecasts:
                        self.super_forecasts[forecast.id] = []
                    self.super_forecasts[forecast.id].append(forecast)
    
    def _load_public_forecasts(self):
        public_forecast_file = self.data_dir / "2024-07-21.ForecastBench.human_public_individual.json"
        if public_forecast_file.exists():
            with open(public_forecast_file, 'r') as f:
                data = json.load(f)
                for f_data in data['forecasts']:
                    forecast = Forecast(**f_data)
                    if forecast.id not in self.public_forecasts:
                        self.public_forecasts[forecast.id] = []
                    self.public_forecasts[forecast.id].append(forecast)

    def get_question(self, question_id: str) -> Optional[Question]:
        return self.questions.get(question_id)
    
    def get_all_questions(self) -> List[Question]:
        return list(self.questions.values())
    
    def search_questions(self, keyword: str) -> List[Question]:
        keyword_lower = keyword.lower()
        results = []
        for question in self.questions.values():
            if (keyword_lower in question.question.lower() or 
                keyword_lower in question.background.lower() or
                keyword_lower in question.resolution_criteria.lower()):
                results.append(question)
        return results
    
    def is_resolved(self, question_id: str) -> bool:
        resolution = self.resolutions.get(question_id)
        return resolution.resolved if resolution else False
    
    def get_resolution(self, question_id: str) -> Optional[Resolution]:
        return self.resolutions.get(question_id)

    def get_super_forecasts(self, question_id: str) -> List[Forecast]:
        return self.super_forecasts.get(question_id, [])

    def get_public_forecasts(self, question_id: str) -> List[Forecast]:
        return self.public_forecasts.get(question_id, [])
    
    def get_question_with_forecasts(self, question_id: str) -> Optional[Dict]:
        question = self.get_question(question_id)
        if not question:
            return None
        
        return {
            'question': question,
            'resolution': self.get_resolution(question_id),
            'super_forecasts': self.get_super_forecasts(question_id),
            'public_forecasts': self.get_public_forecasts(question_id),
            'is_resolved': self.is_resolved(question_id)
        }
    
    def sample_random_question(self) -> Optional[Question]:
        import random
        if self.questions:
            return random.choice(list(self.questions.values()))
        return None
    
    def get_resolved_questions(self) -> List[Question]:
        resolved_ids = {res_id for res_id, res in self.resolutions.items() if res.resolved}
        return [q for q_id, q in self.questions.items() if q_id in resolved_ids]
    
    def get_unresolved_questions(self) -> List[Question]:
        resolved_ids = {res_id for res_id, res in self.resolutions.items() if res.resolved}
        return [q for q_id, q in self.questions.items() if q_id not in resolved_ids]
    
    def get_statistics(self) -> Dict:
        total_questions = len(self.questions)
        resolved_count = len(self.get_resolved_questions())
        unresolved_count = len(self.get_unresolved_questions())

        total_forecasts = sum(len(forecasts) for forecasts in self.super_forecasts.values())
        questions_with_forecasts = len(self.super_forecasts)

        return {
            'total_questions': total_questions,
            'resolved_questions': resolved_count,
            'unresolved_questions': unresolved_count,
            'total_forecasts': total_forecasts,
            'questions_with_forecasts': questions_with_forecasts,
            'average_forecasts_per_question': total_forecasts / questions_with_forecasts if questions_with_forecasts > 0 else 0
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
            'total_forecasts': total_forecasts,
            'forecasts_with_searches': forecasts_with_searches,
            'forecasts_with_urls': forecasts_with_urls,
            'percentage_with_searches': (forecasts_with_searches / total_forecasts * 100) if total_forecasts > 0 else 0,
            'percentage_with_urls': (forecasts_with_urls / total_forecasts * 100) if total_forecasts > 0 else 0,
            'total_searches': total_searches,
            'total_urls': total_urls,
            'avg_searches_when_present': (total_searches / forecasts_with_searches) if forecasts_with_searches > 0 else 0,
            'avg_urls_when_present': (total_urls / forecasts_with_urls) if forecasts_with_urls > 0 else 0,
            'min_searches': min(search_counts) if search_counts else 0,
            'max_searches': max(search_counts) if search_counts else 0,
            'min_urls': min(url_counts) if url_counts else 0,
            'max_urls': max(url_counts) if url_counts else 0
        }


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
        
        super_forecasts = loader.get_super_forecasts(sample.id)
        public_forecasts = loader.get_public_forecasts(sample.id)
        if super_forecasts:
            print(f"  Number of super forecasts: {len(super_forecasts)}")
            print(f"  Average super forecast: {sum(f.forecast for f in super_forecasts) / len(super_forecasts):.3f}")
        if public_forecasts:
            print(f"  Number of public forecasts: {len(public_forecasts)}")
            print(f"  Average public forecast: {sum(f.forecast for f in public_forecasts) / len(public_forecasts):.3f}")