import logging
import math
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import CONFIG, get_api_key


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Labels:
    task: str
    difficulty: str
    private: bool
    needs_web: bool
    source: str = "jev"


_TASK_CRITERIA = {
    "chitchat": "casual conversation or a greeting",
    "rewrite_summarise": "rewrite, rephrase, or summarise provided text",
    "code": "write, fix, or explain code",
    "simple_question": "a short factual question",
    "reasoning": "multi-step reasoning or analysis",
}
_DIFFICULTY_CRITERIA = ["trivial", "easy", "moderate", "hard", "frontier"]


def _answer_value(answer: Any, key: str) -> Any:
    if isinstance(answer, dict):
        return answer[key]
    raise TypeError("Jev answer must be an object")


def label_with_jev(text: str) -> Labels | None:
    config = CONFIG["jev"]
    api_key = get_api_key("OPENCODE_API_KEY")
    if not config["enabled"]:
        return None
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    body = {
        "model": config["model"],
        "state": text,
        "questions": {
            "task": {
                "type": "choice",
                "criteria": _TASK_CRITERIA,
                "instructions": "Which task best describes this request?",
            },
            "difficulty": {
                "type": "score",
                "criteria": _DIFFICULTY_CRITERIA,
                "instructions": "How hard is this request?",
            },
            "private": {
                "type": "noul",
                "instructions": "Does this contain personal data?",
            },
            "needs_web": {
                "type": "noul",
                "instructions": "Does this need fresh information from the web?",
            },
        },
    }

    try:
        response = httpx.post(
            config["url"],
            json=body,
            headers=headers,
            timeout=config["timeout_ms"] / 1000,
        )
        if response.status_code != 200:
            return None

        result = response.json()
        answers = result["answers"]
        task_answer = answers["task"]
        difficulty_answer = answers["difficulty"]
        private_answer = answers["private"]
        needs_web_answer = answers["needs_web"]

        task = _answer_value(task_answer, "choice")
        score = _answer_value(difficulty_answer, "score")
        confidence = _answer_value(difficulty_answer, "confidence")
        private_score = _answer_value(private_answer, "noul")
        needs_web_score = _answer_value(needs_web_answer, "noul")
        if (
            task not in _TASK_CRITERIA
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or isinstance(private_score, bool)
            or not isinstance(private_score, (int, float))
            or isinstance(needs_web_score, bool)
            or not isinstance(needs_web_score, (int, float))
            or not math.isfinite(score)
            or not math.isfinite(confidence)
            or not math.isfinite(private_score)
            or not math.isfinite(needs_web_score)
            or not 0 <= confidence <= 1
            or not 0 <= private_score <= 1
            or not 0 <= needs_web_score <= 1
        ):
            return None

        cutoffs = config["cutoffs"]
        if confidence == cutoffs["difficulty_zero_confidence"]:
            difficulty = "moderate"
        elif score < cutoffs["difficulty_easy"]:
            difficulty = "easy"
        elif score < cutoffs["difficulty_moderate"]:
            difficulty = "moderate"
        else:
            difficulty = "hard"

        return Labels(
            task=task,
            difficulty=difficulty,
            private=private_score >= cutoffs["private"],
            needs_web=needs_web_score >= cutoffs["needs_web"],
        )
    except (
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        OverflowError,
    ) as exc:
        logger.warning("Jev label request failed: %s", type(exc).__name__)
        return None
