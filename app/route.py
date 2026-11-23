import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import CONFIG
from app.label_jev import label_with_jev
from app.rules import matching_rule


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteDecision:
    label: str
    lane: str
    rule: str
    label_source: str
    private: bool = False


def label_with_llm(text: str) -> str:
    labeler = CONFIG["labeler"]
    model_config = CONFIG["models"][labeler["model"]]
    if model_config["kind"] != "local":
        raise ValueError("The labeler model must be local")

    allowed_labels = [
        label for label in CONFIG["labels"] if label != CONFIG["private_label"]
    ]
    labels = ", ".join(allowed_labels)
    response = httpx.post(
        f"{model_config['base_url'].rstrip('/')}/api/chat",
        json={
            "model": model_config["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Choose exactly one label from: {labels}. "
                        "Return only the label."
                    ),
                },
                {"role": "user", "content": text},
            ],
            "stream": False,
        },
        timeout=labeler["timeout_seconds"],
    )
    response.raise_for_status()
    result: Any = response.json()["message"]["content"]
    if not isinstance(result, str):
        raise ValueError("The Ollama label response must be text")
    label = result.strip().strip("\"'").lower()
    if label not in allowed_labels:
        raise ValueError("The Ollama label response was not a configured label")
    return label


def resolve_route(
    text: str,
    private: bool = False,
    router_mode: str = "local",
    local_llm_enabled: bool = True,
) -> RouteDecision:
    jev_failed = False
    if private:
        label = CONFIG["private_label"]
        rule = "private_pattern"
        label_source = "rules"
    else:
        match = matching_rule(text)
        if match:
            label, rule = match
            label_source = "rules"
        else:
            jev_labels = label_with_jev(text) if router_mode == "jev" else None
            if jev_labels is not None:
                label = (
                    CONFIG["private_label"]
                    if jev_labels.private
                    else jev_labels.task
                )
                rule = "jev_label"
                label_source = jev_labels.source
                private = jev_labels.private
            else:
                jev_failed = router_mode == "jev"
                label = None
                if local_llm_enabled:
                    try:
                        label = label_with_llm(text)
                        rule = "llm_label"
                        label_source = "local-llm"
                    except (
                        httpx.HTTPError,
                        json.JSONDecodeError,
                        KeyError,
                        TypeError,
                        ValueError,
                    ) as exc:
                        logger.warning(
                            "Ollama label request failed: %s", type(exc).__name__
                        )
                if label is None:
                    label = CONFIG["default_label"]
                    rule = "default_label"
                    label_source = "default"
                if jev_failed:
                    label_source = f"{label_source} (jev failed)"

    lane = CONFIG["labels"].get(label)
    if lane is None:
        label = CONFIG["default_label"]
        lane = CONFIG["labels"][label]
        rule = "default_label"
        label_source = "default (jev failed)" if jev_failed else "default"
    return RouteDecision(label, lane, rule, label_source, private)


def decide(label: str) -> str:
    return CONFIG["labels"].get(label, CONFIG["labels"][CONFIG["default_label"]])


def select(
    lane: str, private: bool = False, exclude: set[str] | None = None
) -> str | None:
    excluded = exclude or set()
    models = CONFIG["models"]
    private_lane = CONFIG["labels"][CONFIG["private_label"]]
    private_only = private or lane == private_lane
    for name in CONFIG["lanes"].get(lane, []):
        model = models.get(name)
        if not model or not model.get("enabled", False) or name in excluded:
            continue
        if private_only and model.get("kind") != "local":
            continue
        return name
    return None
