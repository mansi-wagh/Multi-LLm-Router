import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.config import CONFIG, get_api_key
from app.models import stream_model
from app.route import RouteDecision, resolve_route, select
from app.rules import looks_private


logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent.parent
UI_FILE = ROOT_DIR / "ui" / "index.html"
RESULT_DIR = ROOT_DIR / "result"
REQUEST_COUNT = 0
MODEL_REQUEST_COUNTS = {name: 0 for name in CONFIG["models"]}
ROUTER_MODE = "local"
LOCAL_LLM_ENABLED = CONFIG["local_llm"]["enabled"]


app = FastAPI(title="llm-router", docs_url=None, redoc_url=None, openapi_url=None)


class ModelToggle(BaseModel):
    model: str
    enabled: bool


class RouterModeUpdate(BaseModel):
    mode: str | None = None
    local_llm: bool | None = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    private_conversation: bool = False


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    models = {
        name: {
            "enabled": model["enabled"],
            "kind": model["kind"],
            "model": model["model"],
        }
        for name, model in CONFIG["models"].items()
    }
    initial_state = {
        "models": models,
        "counters": {
            "total_requests": REQUEST_COUNT,
            "models": MODEL_REQUEST_COUNTS,
        },
        "router_mode": ROUTER_MODE,
        "local_llm_enabled": LOCAL_LLM_ENABLED,
    }
    bootstrap = json.dumps(initial_state).replace("<", "\\u003c")
    page = UI_FILE.read_text(encoding="utf-8").replace("__INITIAL_STATE__", bootstrap)
    return HTMLResponse(page)


@app.post("/models")
def update_model(toggle: ModelToggle) -> dict[str, object]:
    model = CONFIG["models"].get(toggle.model)
    if model is None:
        raise HTTPException(status_code=404, detail="Unknown model")
    model["enabled"] = toggle.enabled
    return {"model": toggle.model, "enabled": model["enabled"]}


@app.post("/router-mode")
def update_router_mode(update: RouterModeUpdate) -> dict[str, object]:
    global ROUTER_MODE, LOCAL_LLM_ENABLED
    if update.mode is not None:
        if update.mode not in {"jev", "local"}:
            raise HTTPException(status_code=422, detail="Mode must be 'jev' or 'local'")
        ROUTER_MODE = update.mode
    if update.local_llm is not None:
        LOCAL_LLM_ENABLED = update.local_llm
    return {"mode": ROUTER_MODE, "local_llm": LOCAL_LLM_ENABLED}


def _event(name: str, data: dict[str, object]) -> str:
    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _route_event(
    decision: RouteDecision, model_name: str | None, is_private: bool
) -> str:
    return _event(
        "route",
        {
            "lane": decision.lane,
            "model": model_name,
            "rule": decision.rule,
            "label_source": decision.label_source,
            "private": is_private,
            "counters": {
                "total_requests": REQUEST_COUNT,
                "models": MODEL_REQUEST_COUNTS,
            },
        },
    )


def _save_chat(record: dict[str, Any]) -> bool:
    try:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULT_DIR / f"{record['id']}.json"
        temporary_path = RESULT_DIR / f".{record['id']}.tmp"
        temporary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return True
    except OSError:
        logger.exception("Could not save chat response")
        return False


async def _chat_events(request: ChatRequest) -> AsyncIterator[str]:
    global REQUEST_COUNT
    REQUEST_COUNT += 1

    messages = [message.model_dump() for message in request.messages]
    record: dict[str, Any] = {
        "id": uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "messages": messages,
        "route": None,
        "models_used": [],
        "assistant": "",
        "status": "pending",
        "error": None,
    }
    latest_user_text = next(
        message["content"]
        for message in reversed(messages)
        if message["role"] == "user"
    )
    private_text = "\n".join(message["content"] for message in messages)
    private = looks_private(private_text) or request.private_conversation
    decision = await run_in_threadpool(
        resolve_route,
        latest_user_text,
        private,
        ROUTER_MODE,
        LOCAL_LLM_ENABLED,
    )
    private = private or decision.private
    is_private = (
        private
        or decision.lane
        == CONFIG["labels"][CONFIG["private_label"]]
    )
    model_name = select(decision.lane, is_private)
    record["route"] = {
        "lane": decision.lane,
        "model": model_name,
        "rule": decision.rule,
        "label_source": decision.label_source,
        "private": is_private,
    }
    if model_name is not None:
        MODEL_REQUEST_COUNTS[model_name] += 1
    yield _route_event(decision, model_name, is_private)

    if model_name is None:
        message = (
            "No enabled local model is available for this private request"
            if is_private
            else f"No enabled model is available for lane '{decision.lane}'"
        )
        record["status"] = "failed"
        record["error"] = message
        if not _save_chat(record):
            yield _event("error", {"message": "Could not save chat response"})
        yield _event("error", {"message": message})
        return

    excluded: set[str] = set()
    while model_name is not None:
        received_token = False
        record["models_used"].append(model_name)
        try:
            model_config: dict[str, Any] = CONFIG["models"][model_name]
            async for token in stream_model(
                model_name,
                messages,
                model_config,
                get_api_key(model_config.get("api_key_env")),
            ):
                received_token = True
                record["assistant"] += token
                yield _event("token", {"model": model_name, "token": token})
            if received_token:
                record["status"] = "completed"
                record["route"]["final_model"] = model_name
                if not _save_chat(record):
                    yield _event("error", {"message": "Could not save chat response"})
                yield _event("done", {"model": model_name})
                return
            logger.warning("Model %s returned no tokens", model_name)
        except (
            httpx.HTTPError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            logger.warning(
                "Model %s stream failed: %s", model_name, type(exc).__name__
            )
            if received_token:
                record["status"] = "failed"
                record["route"]["final_model"] = model_name
                record["error"] = f"Model '{model_name}' failed after streaming started"
                if not _save_chat(record):
                    yield _event("error", {"message": "Could not save chat response"})
                yield _event(
                    "error",
                    {"message": f"Model '{model_name}' failed after streaming started"},
                )
                return

        excluded.add(model_name)
        model_name = select(decision.lane, is_private, excluded)
        if model_name is None:
            if is_private:
                message = "All enabled local models failed before returning a response"
            else:
                message = "All enabled models in this lane failed before returning a response"
            record["status"] = "failed"
            record["error"] = message
            if not _save_chat(record):
                yield _event("error", {"message": "Could not save chat response"})
            yield _event("error", {"message": message})
            return

        if record["route"]["model"] != model_name:
            record["route"]["fallback_model"] = model_name
        MODEL_REQUEST_COUNTS[model_name] += 1
        yield _route_event(decision, model_name, is_private)


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    if not request.messages or not any(
        message.role == "user" for message in request.messages
    ):
        raise HTTPException(
            status_code=422, detail="At least one user message is required"
        )
    return StreamingResponse(
        _chat_events(request),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
