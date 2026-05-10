"""Application service layer for API handlers."""

from __future__ import annotations

import base64
import json
import logging
import time

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.application.chat_service import cleanup_files, normalize_chat_request, normalize_gemini_request, normalize_openai_tools
from aistudio_api.domain.errors import AistudioError, AuthError, RequestError, UsageLimitExceeded
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart
from aistudio_api.api.responses import (
    chat_completion_response,
    new_chat_id,
    sse_chunk,
    sse_error,
    sse_usage_chunk,
    to_gemini_parts,
    to_gemini_usage_metadata,
    to_openai_tool_calls,
)
from aistudio_api.api.schemas import ChatRequest, GeminiGenerateContentRequest, ImageRequest
from aistudio_api.api.state import runtime_state

logger = logging.getLogger("aistudio.server")


async def _try_switch_account() -> bool:
    """尝试切换到下一个可用账号。返回是否成功切换。"""
    rotator = runtime_state.rotator
    if rotator is None:
        return False

    # 获取下一个账号
    next_account = await rotator.get_next_account()
    if next_account is None:
        return False

    account_service = runtime_state.account_service
    client = runtime_state.client

    if not all([account_service, client]):
        return False

    # 切换账号时清掉 snapshot，避免复用旧页面态。
    result = await account_service.activate_account(
        next_account.id,
        client._session,
        runtime_state.snapshot_cache,
        None,  # skip lock — caller already holds it
        keep_snapshot_cache=False,
    )
    return result is not None


def health_response() -> dict:
    busy_lock = runtime_state.busy_lock
    return {"status": "ok", "busy": busy_lock.locked() if busy_lock else False}


def stats_response() -> dict:
    stats = dict(runtime_state.model_stats)
    totals = {
        "requests": sum(s["requests"] for s in stats.values()),
        "success": sum(s["success"] for s in stats.values()),
        "rate_limited": sum(s["rate_limited"] for s in stats.values()),
        "errors": sum(s["errors"] for s in stats.values()),
        "prompt_tokens": sum(s["prompt_tokens"] for s in stats.values()),
        "completion_tokens": sum(s["completion_tokens"] for s in stats.values()),
        "total_tokens": sum(s["total_tokens"] for s in stats.values()),
    }
    return {"models": stats, "totals": totals}


async def handle_chat(req: ChatRequest, client: AIStudioClient):
    busy_lock = runtime_state.busy_lock
    if busy_lock is None:
        raise HTTPException(503, detail={"message": "Server not ready", "type": "service_unavailable"})
    if busy_lock.locked():
        raise HTTPException(429, detail={"message": "Server is busy", "type": "rate_limit_exceeded"})

    max_retries = 3  # 最多重试次数
    last_error = None

    for attempt in range(max_retries):
        async with busy_lock:
            # 首次尝试时，仅在没有活跃账号时才轮询（避免每次请求都重建浏览器）
            if attempt == 0:
                account_svc = runtime_state.account_service
                if account_svc and not account_svc.get_active_account():
                    await _try_switch_account()
            normalized = normalize_chat_request(req.messages, req.model)
            model = normalized["model"]
            tmp_files = list(normalized["cleanup_paths"])

            try:
                logger.info(
                    "Chat: model=%s, contents=%s, capture_prompt=%s..., images=%s, stream=%s, attempt=%d",
                    model,
                    len(normalized["contents"]),
                    normalized["capture_prompt"][:50],
                    len(normalized["capture_images"]),
                    req.stream,
                    attempt + 1,
                )
                tools = normalize_openai_tools(req.tools)

                # Gemma 4 默认开启 Google Search
                if tools is None and any(m in model for m in ("gemma-4-26b-a4b-it", "gemma-4-31b-it")):
                    from aistudio_api.infrastructure.gateway.request_rewriter import TOOLS_TEMPLATES
                    tools = [TOOLS_TEMPLATES["google_search"]]

                if req.stream:
                    include_usage = True
                    if req.stream_options is not None:
                        include_usage = req.stream_options.include_usage
                    return _build_streaming_response(
                        client=client,
                        capture_prompt=normalized["capture_prompt"],
                        model=model,
                        capture_images=normalized["capture_images"] if normalized["capture_images"] else None,
                        contents=normalized["contents"],
                        system_instruction=normalized["system_instruction"],
                        cleanup_paths=tmp_files,
                        include_usage=include_usage,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        max_tokens=req.max_tokens,
                        tools=tools,
                    )

                output = await client.generate_content(
                    model=model,
                    capture_prompt=normalized["capture_prompt"],
                    capture_images=normalized["capture_images"] if normalized["capture_images"] else None,
                    contents=normalized["contents"],
                    system_instruction_content=(
                        AistudioContent(role="user", parts=[AistudioPart(text=normalized["system_instruction"])])
                        if normalized["system_instruction"]
                        else None
                    ),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                    max_tokens=req.max_tokens,
                    tools=tools,
                    sanitize_plain_text=True,
                )

                # 记录成功
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_success(account.id)

                runtime_state.record(model, "success", output.usage)
                return chat_completion_response(
                    model=model,
                    content=output.text,
                    thinking=output.thinking,
                    usage=output.usage,
                    function_calls=output.function_calls,
                )
            except UsageLimitExceeded as exc:
                runtime_state.record(model, "rate_limited")
                last_error = exc

                # 记录限流
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_rate_limited(account.id)

                # 尝试切换账号
                if await _try_switch_account():
                    logger.info("429 限流，已切换账号，重试 %d/%d", attempt + 1, max_retries)
                    continue
                else:
                    logger.warning("429 限流，无法切换账号")
                    raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model, "errors")
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_error(account.id)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model, "errors")
                logger.error("Chat error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            finally:
                if not req.stream:
                    cleanup_files(tmp_files)

    # 所有重试都失败
    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


async def handle_image_generation(req: ImageRequest, client: AIStudioClient):
    busy_lock = runtime_state.busy_lock
    if busy_lock is None:
        raise HTTPException(503, detail={"message": "Server not ready", "type": "service_unavailable"})
    if busy_lock.locked():
        raise HTTPException(429, detail={"message": "Server is busy", "type": "rate_limit_exceeded"})

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        async with busy_lock:
            if attempt == 0:
                account_svc = runtime_state.account_service
                if account_svc and not account_svc.get_active_account():
                    await _try_switch_account()
            try:
                logger.info("Image: model=%s, prompt=%s..., attempt=%d", req.model, req.prompt[:50], attempt + 1)
                output = await client.generate_image(prompt=req.prompt, model=req.model, size=req.size, google_search=req.google_search)

                data = []
                for img in output.images:
                    b64 = base64.b64encode(img.data).decode("ascii")
                    data.append({"b64_json": b64, "revised_prompt": output.text or ""})

                # 记录成功
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_success(account.id)

                runtime_state.record(req.model, "success", output.usage)
                return {"created": int(time.time()), "data": data}
            except UsageLimitExceeded as exc:
                runtime_state.record(req.model, "rate_limited")
                last_error = exc

                # 记录限流
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_rate_limited(account.id)

                # 尝试切换账号
                if await _try_switch_account():
                    logger.info("Image 429 限流，已切换账号，重试 %d/%d", attempt + 1, max_retries)
                    continue
                else:
                    logger.warning("Image 429 限流，无法切换账号")
                    raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(req.model, "errors")
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_error(account.id)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(req.model, "errors")
                logger.error("Image error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc

    # 所有重试都失败
    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


def _build_streaming_response(
    *,
    client: AIStudioClient,
    capture_prompt: str,
    model: str,
    capture_images: list[str] | None,
    contents: list[AistudioContent],
    system_instruction: str | None,
    cleanup_paths: list[str],
    include_usage: bool = False,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    tools: list[list] | None = None,
) -> StreamingResponse:
    async def stream_response():
        busy_lock = runtime_state.busy_lock
        if busy_lock is None:
            yield sse_error("Server not ready")
            cleanup_files(cleanup_paths)
            return

        async with busy_lock:
            try:
                chat_id = new_chat_id()
                final_usage = None
                saw_tool_calls = False
                for stream_attempt in range(2):
                    try:
                        async for event_type, text in client.stream_generate_content(
                            model=model,
                            capture_prompt=capture_prompt,
                            capture_images=capture_images,
                            contents=contents,
                            system_instruction_content=(
                                AistudioContent(role="user", parts=[AistudioPart(text=system_instruction)])
                                if system_instruction
                                else None
                            ),
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            tools=tools,
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            if event_type == "body" and text:
                                yield sse_chunk(chat_id, model, text, include_usage=include_usage)
                            elif event_type == "thinking" and text:
                                yield sse_chunk(chat_id, model, "", thinking=text, include_usage=include_usage)
                            elif event_type == "tool_calls" and text:
                                saw_tool_calls = True
                                yield sse_chunk(
                                    chat_id,
                                    model,
                                    "",
                                    tool_calls=to_openai_tool_calls(text if isinstance(text, list) else []),
                                    include_usage=include_usage,
                                )
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        break
                    except RequestError as exc:
                        if exc.status == 204 and stream_attempt == 0:
                            logger.warning("Stream 收到 204，清理 snapshot 缓存后重试一次")
                            client.clear_snapshot_cache()
                            continue
                        raise
                    except AuthError as exc:
                        if stream_attempt == 0:
                            logger.warning("Stream 鉴权异常，清理 snapshot 缓存后重试一次: %s", exc)
                            client.clear_snapshot_cache()
                            continue
                        raise

                runtime_state.record(model, "success", final_usage)
                yield sse_chunk(chat_id, model, "", finish="tool_calls" if saw_tool_calls else "stop", include_usage=include_usage)
                if include_usage:
                    yield sse_usage_chunk(chat_id, model, final_usage)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Stream error: %s", exc, exc_info=True)
                runtime_state.record(model, "errors")
                yield sse_error(str(exc))
            finally:
                cleanup_files(cleanup_paths)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def handle_gemini_generate_content(
    model_path: str,
    req: GeminiGenerateContentRequest,
    client: AIStudioClient,
    *,
    stream: bool,
):
    busy_lock = runtime_state.busy_lock
    if busy_lock is None:
        raise HTTPException(503, detail={"message": "Server not ready", "type": "service_unavailable"})
    if busy_lock.locked():
        raise HTTPException(429, detail={"message": "Server is busy", "type": "rate_limit_exceeded"})

    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        async with busy_lock:
            if attempt == 0:
                account_svc = runtime_state.account_service
                if account_svc and not account_svc.get_active_account():
                    await _try_switch_account()
            normalized = None
            try:
                normalized = normalize_gemini_request(req, model_path)
                logger.info(
                    "Gemini: model=%s, contents=%s, stream=%s, attempt=%d",
                    normalized["model"],
                    len(req.contents),
                    stream,
                    attempt + 1,
                )

                if stream:
                    return _build_gemini_streaming_response(client=client, normalized=normalized)

                output = await client.generate_content(
                    model=normalized["model"],
                    capture_prompt=normalized["capture_prompt"],
                    capture_images=normalized["capture_images"],
                    contents=normalized["contents"],
                    system_instruction_content=normalized["system_instruction"],
                    tools=normalized["tools"],
                    temperature=normalized["temperature"],
                    top_p=normalized["top_p"],
                    top_k=normalized["top_k"],
                    max_tokens=normalized["max_tokens"],
                    generation_config_overrides=normalized["generation_config_overrides"],
                    sanitize_plain_text=False,
                )

                # 记录成功
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_success(account.id)

                runtime_state.record(normalized["model"], "success", output.usage)
                return {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": to_gemini_parts(
                                    output.text,
                                    function_calls=output.function_calls,
                                    function_responses=output.function_responses,
                                    thinking=output.thinking,
                                ),
                            },
                            "finishReason": "STOP" if not output.function_calls else "FUNCTION_CALL",
                        }
                    ],
                    "usageMetadata": to_gemini_usage_metadata(output.usage),
                }
            except ValueError as exc:
                raise HTTPException(400, detail={"message": str(exc), "type": "bad_request"}) from exc
            except UsageLimitExceeded as exc:
                runtime_state.record(model_path, "rate_limited")
                last_error = exc

                # 记录限流
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_rate_limited(account.id)

                # 尝试切换账号
                if await _try_switch_account():
                    logger.info("Gemini 429 限流，已切换账号，重试 %d/%d", attempt + 1, max_retries)
                    continue
                else:
                    logger.warning("Gemini 429 限流，无法切换账号")
                    raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model_path, "errors")
                rotator = runtime_state.rotator
                if rotator:
                    account = runtime_state.account_service.get_active_account() if runtime_state.account_service else None
                    if account:
                        rotator.record_error(account.id)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model_path, "errors")
                logger.error("Gemini error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            finally:
                if normalized is not None and not stream:
                    cleanup_files(normalized["cleanup_paths"])

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


def _build_gemini_streaming_response(*, client: AIStudioClient, normalized: dict) -> StreamingResponse:
    async def stream_response():
        busy_lock = runtime_state.busy_lock
        if busy_lock is None:
            yield "data: " + json.dumps({"error": {"message": "Server not ready"}}, ensure_ascii=False) + "\n\n"
            cleanup_files(normalized["cleanup_paths"])
            return

        async with busy_lock:
            try:
                final_usage = None
                for stream_attempt in range(2):
                    try:
                        async for event_type, text in client.stream_generate_content(
                            model=normalized["model"],
                            capture_prompt=normalized["capture_prompt"],
                            capture_images=normalized["capture_images"],
                            contents=normalized["contents"],
                            system_instruction_content=normalized["system_instruction"],
                            tools=normalized["tools"],
                            temperature=normalized["temperature"],
                            top_p=normalized["top_p"],
                            top_k=normalized["top_k"],
                            max_tokens=normalized["max_tokens"],
                            generation_config_overrides=normalized["generation_config_overrides"],
                            sanitize_plain_text=False,
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            if event_type == "body" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {"role": "model", "parts": [{"text": text}]},
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "thinking" and text:
                                yield "data: " + json.dumps(
                                    {
                                        "candidates": [
                                            {
                                                "content": {
                                                    "role": "model",
                                                    "parts": [{"text": text, "thought": True}],
                                                },
                                                "finishReason": None,
                                            }
                                        ]
                                    },
                                    ensure_ascii=False,
                                ) + "\n\n"
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        break
                    except RequestError as exc:
                        if exc.status == 204 and stream_attempt == 0:
                            logger.warning("Gemini stream 收到 204，清理 snapshot 缓存后重试一次")
                            client.clear_snapshot_cache()
                            continue
                        raise
                    except AuthError as exc:
                        if stream_attempt == 0:
                            logger.warning("Gemini stream 鉴权异常，清理 snapshot 缓存后重试一次: %s", exc)
                            client.clear_snapshot_cache()
                            continue
                        raise

                runtime_state.record(normalized["model"], "success", final_usage)
                if final_usage:
                    yield "data: " + json.dumps(
                        {
                            "candidates": [],
                            "usageMetadata": to_gemini_usage_metadata(final_usage),
                        },
                        ensure_ascii=False,
                    ) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Gemini stream error: %s", exc, exc_info=True)
                runtime_state.record(normalized["model"], "errors")
                yield "data: " + json.dumps({"error": {"message": str(exc)}}, ensure_ascii=False) + "\n\n"
            finally:
                cleanup_files(normalized["cleanup_paths"])

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
