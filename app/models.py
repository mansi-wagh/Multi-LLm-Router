import json
from collections.abc import AsyncIterator, Mapping, Sequence

import httpx


async def _stream_openai_compatible(
    base_url: str,
    api_key: str | None,
    model: str,
    messages: Sequence[Mapping[str, str]],
) -> AsyncIterator[str]:
    if not api_key:
        raise ValueError(f"No API key configured for model {model}")

    url = f"{base_url.rstrip('/')}/chat/completions"
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                if not data:
                    continue

                payload = json.loads(data)
                choices = payload.get("choices", [])
                if choices:
                    token = choices[0].get("delta", {}).get("content")
                    if token:
                        yield token


async def stream_model(
    model_name: str,
    messages: Sequence[Mapping[str, str]],
    model_config: Mapping[str, object],
    api_key: str | None = None,
) -> AsyncIterator[str]:
    if model_config["kind"] == "local":
        url = f"{str(model_config['base_url']).rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                url,
                json={
                    "model": model_config["model"],
                    "messages": messages,
                    "stream": True,
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    payload = json.loads(line)
                    if payload.get("error"):
                        raise ValueError(f"{model_name} returned an error")
                    token = payload.get("message", {}).get("content")
                    if token:
                        yield token
                    if payload.get("done"):
                        return
        return

    async for token in _stream_openai_compatible(
        str(model_config["base_url"]),
        api_key,
        str(model_config["model"]),
        messages,
    ):
        yield token
