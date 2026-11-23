import asyncio
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


ENDPOINT = "https://opencode.ai/zen/v1/systemone"
STATE = "My email is test@example.com and I need help fixing a Python bug"
TIMEOUT_SECONDS = 10
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.is_file():
    load_dotenv(ENV_FILE)

API_KEY = os.getenv("OPENCODE_API_KEY")


def send_request(
    label: str, body: dict[str, Any], api_key: str | None = None
) -> httpx.Response | None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.perf_counter()
    try:
        response = httpx.post(
            ENDPOINT,
            json=body,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.RequestError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        print(f"\n{label}")
        print("Status code: unavailable")
        print(f"Raw response: request failed ({type(exc).__name__}): {exc}")
        print(f"Latency: {latency_ms:.2f} ms")
        return None

    latency_ms = (time.perf_counter() - started) * 1000
    print(f"\n{label}")
    print(f"Status code: {response.status_code}")
    print("Raw response:")
    print(response.text)
    print(f"Latency: {latency_ms:.2f} ms")
    return response


def send_with_auth_fallback(
    label: str, body: dict[str, Any]
) -> tuple[httpx.Response | None, str | None]:
    response = send_request(label, body)
    if response is not None and response.status_code in {401, 403}:
        if not API_KEY:
            print("OPENCODE_API_KEY is missing; cannot retry with authorization.")
            return response, None
        response = send_request(f"{label} (authenticated retry)", body, API_KEY)
        return response, API_KEY
    return response, None


def needs_delayed_retry(response: httpx.Response | None) -> bool:
    return response is not None and (
        response.status_code == 422
        or "endpoint is unavailable" in response.text.lower()
    )


def run_question(
    label: str, question_name: str, question: dict[str, Any]
) -> None:
    body = {
        "model": "jev-1.13-free",
        "state": STATE,
        "questions": {question_name: question},
    }
    response, auth_key = send_with_auth_fallback(label, body)
    if needs_delayed_retry(response):
        print(f"\n{label} failed; retrying once in 20 seconds.")
        time.sleep(20)
        send_request(f"{label} (retry)", body, auth_key)


async def send_questions_in_parallel(
    questions: dict[str, dict[str, str]], api_key: str | None
) -> None:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        responses = await asyncio.gather(
            *(
                client.post(
                    ENDPOINT,
                    json={
                        "model": "jev-1.13-free",
                        "state": STATE,
                        "questions": {name: question},
                    },
                    headers=headers,
                )
                for name, question in questions.items()
            ),
            return_exceptions=True,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000

    for (name, _), result in zip(questions.items(), responses):
        print(f"\nParallel fallback: {name}")
        if isinstance(result, httpx.Response):
            print(f"Status code: {result.status_code}")
            print("Raw response:")
            print(result.text)
        else:
            print("Status code: unavailable")
            print(f"Raw response: request failed ({type(result).__name__}): {result}")
    print(f"\nParallel fallback total time: {elapsed_ms:.2f} ms")


def main() -> None:
    calls = [
        (
            "Call 1: needs_web",
            "needs_web",
            {
                "type": "noul",
                "instructions": "Does this need fresh information from the web?",
            },
        ),
        (
    "Call 2: difficulty",
    "difficulty",
    {
        "type": "score",
        "instructions": "How hard is this request for an AI model to answer well?",
        "criteria": ["trivial", "easy", "moderate", "hard", "frontier"],
    },
        ),
        (
    "Call 3: task",
    "task",
    {
        "type": "choice",
        "instructions": "Which task best describes this request?",
        "criteria": {
            "chitchat": "casual conversation or greeting",
            "rewrite_summarise": "rewrite, rephrase or summarise given text",
            "code": "write, fix or explain code",
            "simple_question": "a short factual question",
            "reasoning": "multi-step reasoning or analysis",
        },
    },
        ),
    ]

    for index, (label, name, question) in enumerate(calls):
        run_question(label, name, question)
        if index < len(calls) - 1:
            time.sleep(3)

    run_question("Final check: needs_web again", calls[0][1], calls[0][2])

    batch_questions = {
        "is_code": {
            "type": "noul",
            "instructions": "Does this request involve writing, fixing, or explaining code? Answer yes or no.",
        },
        "is_chitchat": {
            "type": "noul",
            "instructions": "Is this request casual conversation or a greeting? Answer yes or no.",
        },
        "needs_reasoning": {
            "type": "noul",
            "instructions": "Does answering this request require multi-step reasoning or analysis? Answer yes or no.",
        },
        "is_hard": {
            "type": "noul",
            "instructions": "Is this request difficult for an AI model to answer well? Answer yes or no.",
        },
        "private": {
            "type": "noul",
            "instructions": "Does this text contain personal data? Answer yes or no.",
        },
        "needs_web": {
            "type": "noul",
            "instructions": "Does answering this request require fresh information from the web? Answer yes or no.",
        },
    }
    batch_body = {
        "model": "jev-1.13-free",
        "state": STATE,
        "questions": batch_questions,
    }
    response, auth_key = send_with_auth_fallback(
        "Final call: six questions together", batch_body
    )
    if response is None or not response.is_success:
        print("\nCombined request failed; sending six separate requests in parallel.")
        asyncio.run(send_questions_in_parallel(batch_questions, auth_key or API_KEY))


if __name__ == "__main__":
    main()
