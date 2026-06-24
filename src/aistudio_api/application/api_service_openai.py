"""OpenAI-compatible application service handlers."""

from __future__ import annotations

import json
import re

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from aistudio_api.api.responses import (
    chat_completion_response,
    new_chat_id,
    sse_chunk,
    sse_error,
    sse_usage_chunk,
    to_openai_tool_calls,
)
from aistudio_api.api.schemas import ChatRequest, ImageRequest
from aistudio_api.config import settings
from aistudio_api.api.state import runtime_state
from aistudio_api.application.api_service_common import (
    MAX_RETRIES,
    build_inline_image_parts,
    ensure_active_account,
    image_response,
    logger,
    record_rotator_event,
    require_busy_lock,
    try_switch_account,
    validate_image_request_options,
)
from aistudio_api.application.chat_service import cleanup_files, normalize_chat_request, normalize_openai_tools
from aistudio_api.domain.errors import AistudioError, AuthError, RequestError, UsageLimitExceeded
from aistudio_api.infrastructure.gateway.client import AIStudioClient
from aistudio_api.infrastructure.gateway.model_defaults import resolve_model_defaults
from aistudio_api.infrastructure.gateway.wire_types import (
    AistudioContent,
    AistudioPart,
    AistudioThinkingConfig,
    ThinkingLevel,
)


async def handle_chat(req: ChatRequest, client: AIStudioClient):
    busy_lock = require_busy_lock()
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            normalized = normalize_chat_request(req.messages, req.model)
            model = normalized["model"]
            tmp_files = list(normalized["cleanup_paths"])

            try:
                # Disable implicit/default tools only for the immediate follow-up
                # turn after a tool result.  Explicit OpenAI tools must survive so
                # OpenAI-compatible agent clients can decide whether to call
                # another tool or finish.
                pending_tool_result = _last_non_system_role(req.messages) == "tool"
                request_options = _resolve_openai_request_options(req)
                tools = None if req.tools is None else (normalize_openai_tools(req.tools) or [])

                if (
                    req.tools is None
                    and not pending_tool_result
                    and not _is_context_summarization_request(normalized["capture_prompt"])
                ):
                    from aistudio_api.infrastructure.gateway.request_rewriter import build_tools_from_names

                    model_defaults = resolve_model_defaults(model)
                    default_tool_names = list(model_defaults.default_tools or [])
                    if request_options["google_search"] is True and "google_search" not in default_tool_names:
                        default_tool_names.append("google_search")
                    elif request_options["google_search"] is False:
                        default_tool_names = _remove_google_search_tool_names(default_tool_names)
                    if default_tool_names:
                        tools = build_tools_from_names(
                            default_tool_names,
                            model=model,
                            is_image_model=model_defaults.is_image_model,
                        )
                elif req.tools is not None and _request_field_set(req, "google_search") and req.google_search:
                    from aistudio_api.infrastructure.gateway.request_rewriter import build_tools_from_names

                    model_defaults = resolve_model_defaults(model)
                    tools = list(tools or [])
                    tools.extend(
                        build_tools_from_names(
                            ["google_search"],
                            model=model,
                            is_image_model=model_defaults.is_image_model,
                        )
                    )

                logger.info(
                    "Chat: model=%s, contents=%s, capture_prompt=%s..., images=%s, stream=%s, attempt=%d, last_role=%s, req_tools=%s, forwarded_tools=%s",
                    model,
                    len(normalized["contents"]),
                    normalized["capture_prompt"][:50],
                    len(normalized["capture_images"]),
                    req.stream,
                    attempt + 1,
                    "tool" if pending_tool_result else _last_non_system_role(req.messages),
                    len(req.tools or []),
                    "none" if tools is None else len(tools),
                )

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
                        temperature=request_options["temperature"],
                        top_p=request_options["top_p"],
                        top_k=request_options["top_k"],
                        max_tokens=request_options["max_tokens"],
                        tools=tools,
                        safety_settings=request_options["safety_settings"],
                        generation_config_overrides=request_options["generation_config_overrides"],
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
                    temperature=request_options["temperature"],
                    top_p=request_options["top_p"],
                    top_k=request_options["top_k"],
                    max_tokens=request_options["max_tokens"],
                    tools=tools,
                    safety_settings=request_options["safety_settings"],
                    generation_config_overrides=request_options["generation_config_overrides"],
                    sanitize_plain_text=True,
                )

                record_rotator_event("success")
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

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                logger.error("Chat error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            finally:
                if not req.stream:
                    cleanup_files(tmp_files)

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


async def handle_image_generation(req: ImageRequest, client: AIStudioClient):
    validate_image_request_options(size=req.size, n=req.n)

    busy_lock = require_busy_lock()
    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            try:
                logger.info("Image: model=%s, prompt=%s..., attempt=%d", req.model, req.prompt[:50], attempt + 1)
                output = await client.generate_image(
                    prompt=req.prompt,
                    model=req.model,
                    size=req.size,
                    google_search=req.google_search,
                    image_search=req.image_search,
                    use_default_tools=not bool({"google_search", "image_search"} & req.model_fields_set),
                )
                record_rotator_event("success")
                runtime_state.record(req.model, "success", output.usage)
                return image_response(output)
            except UsageLimitExceeded as exc:
                runtime_state.record(req.model, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("Image 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("Image 429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(req.model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(req.model, "errors")
                record_rotator_event("error")
                logger.error("Image error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc

    raise HTTPException(429, detail={"message": str(last_error), "type": "rate_limit_exceeded"}) from last_error


async def handle_image_edit(
    prompt: str,
    image_files: list,
    mask_file,
    model: str,
    n: int,
    size: str,
    client: AIStudioClient,
):
    validate_image_request_options(size=size, n=n)

    busy_lock = require_busy_lock()
    image_parts = await build_inline_image_parts(image_files)
    request_contents = [AistudioContent(role="user", parts=[*image_parts, AistudioPart(text=prompt)])]

    last_error = None

    for attempt in range(MAX_RETRIES):
        async with busy_lock:
            await ensure_active_account(attempt)
            try:
                logger.info(
                    "Image Edit: model=%s, prompt=%s..., images=%d, attempt=%d",
                    model,
                    prompt[:50],
                    len(image_parts),
                    attempt + 1,
                )
                output = await client.generate_image(
                    prompt=prompt,
                    model=model,
                    size=size,
                    contents=request_contents,
                )
                record_rotator_event("success")
                runtime_state.record(model, "success", output.usage)
                return image_response(output)
            except UsageLimitExceeded as exc:
                runtime_state.record(model, "rate_limited")
                last_error = exc

                record_rotator_event("rate_limited")
                if await try_switch_account():
                    logger.info("Image Edit 429 限流，已切换账号，重试 %d/%d", attempt + 1, MAX_RETRIES)
                    continue
                logger.warning("Image Edit 429 限流，无法切换账号")
                raise HTTPException(429, detail={"message": str(exc), "type": "rate_limit_exceeded"}) from exc
            except AistudioError as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc
            except Exception as exc:
                runtime_state.record(model, "errors")
                record_rotator_event("error")
                logger.error("Image Edit error: %s", exc, exc_info=True)
                raise HTTPException(500, detail={"message": str(exc), "type": "server_error"}) from exc

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
    safety_settings: list[list] | None = None,
    generation_config_overrides: dict | None = None,
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
                buffered_body: list[str] = []
                for stream_attempt in range(MAX_RETRIES):
                    try:
                        has_yielded_data = False
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
                            safety_settings=safety_settings,
                            generation_config_overrides=generation_config_overrides,
                            force_refresh_capture=stream_attempt > 0,
                        ):
                            has_yielded_data = True
                            if event_type == "body" and text:
                                if tools:
                                    buffered_body.append(str(text))
                                else:
                                    yield sse_chunk(chat_id, model, text, include_usage=include_usage)
                            elif event_type == "thinking" and text:
                                yield sse_chunk(chat_id, model, "", thinking=text, include_usage=include_usage)
                            elif event_type == "tool_calls" and text:
                                saw_tool_calls = True
                                tool_names = [
                                    str(call.get("name") or call.get("function_name") or "")
                                    for call in (text if isinstance(text, list) else [])
                                    if isinstance(call, dict)
                                ]
                                logger.info(
                                    "OpenAI stream tool_calls: model=%s, count=%d, names=%s",
                                    model,
                                    len(tool_names),
                                    tool_names[:10],
                                )
                                yield sse_chunk(
                                    chat_id,
                                    model,
                                    "",
                                    tool_calls=to_openai_tool_calls(text if isinstance(text, list) else [], include_index=True),
                                    include_usage=include_usage,
                                )
                            elif event_type == "usage":
                                final_usage = text if isinstance(text, dict) else None
                        break
                    except UsageLimitExceeded:
                        runtime_state.record(model, "rate_limited")
                        record_rotator_event("rate_limited")
                        if not has_yielded_data and stream_attempt < MAX_RETRIES - 1 and await try_switch_account():
                            logger.warning("Stream 429 限流，已切换账号，重试 %d/%d", stream_attempt + 1, MAX_RETRIES)
                            continue
                        raise
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

                record_rotator_event("success")
                runtime_state.record(model, "success", final_usage)
                if tools and buffered_body and not saw_tool_calls:
                    body_text = "".join(buffered_body)
                    pseudo_tool_calls = _extract_pseudo_tool_calls(body_text)
                    if pseudo_tool_calls:
                        saw_tool_calls = True
                        logger.info(
                            "OpenAI stream pseudo tool_calls converted: model=%s, count=%d, names=%s",
                            model,
                            len(pseudo_tool_calls),
                            [str(call.get("name") or "") for call in pseudo_tool_calls[:10]],
                        )
                        yield sse_chunk(
                            chat_id,
                            model,
                            "",
                            tool_calls=to_openai_tool_calls(pseudo_tool_calls, include_index=True),
                            include_usage=include_usage,
                        )
                    else:
                        continuation_text, replace_buffered_body = await _maybe_continue_incomplete_final_text(
                            client=client,
                            model=model,
                            capture_prompt=capture_prompt,
                            capture_images=capture_images,
                            contents=contents,
                            system_instruction=system_instruction,
                            partial_text=body_text,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            max_tokens=max_tokens,
                            safety_settings=safety_settings,
                            generation_config_overrides=generation_config_overrides,
                        )
                        if continuation_text:
                            if replace_buffered_body:
                                buffered_body = [continuation_text]
                            else:
                                buffered_body.append(continuation_text)
                            continued_body_text = "".join(buffered_body)
                            pseudo_tool_calls = [] if replace_buffered_body else _extract_pseudo_tool_calls(continued_body_text)
                            if pseudo_tool_calls:
                                saw_tool_calls = True
                                logger.info(
                                    "OpenAI stream pseudo tool_calls converted after continuation: model=%s, count=%d, names=%s",
                                    model,
                                    len(pseudo_tool_calls),
                                    [str(call.get("name") or "") for call in pseudo_tool_calls[:10]],
                                )
                                yield sse_chunk(
                                    chat_id,
                                    model,
                                    "",
                                    tool_calls=to_openai_tool_calls(pseudo_tool_calls, include_index=True),
                                    include_usage=include_usage,
                                )
                        if not saw_tool_calls:
                            for body_chunk in buffered_body:
                                yield sse_chunk(chat_id, model, body_chunk, include_usage=include_usage)
                logger.info(
                    "OpenAI stream finish: model=%s, finish_reason=%s, final_usage=%s",
                    model,
                    "tool_calls" if saw_tool_calls else "stop",
                    final_usage,
                )
                yield sse_chunk(chat_id, model, "", finish="tool_calls" if saw_tool_calls else "stop", include_usage=include_usage)
                if include_usage:
                    yield sse_usage_chunk(chat_id, model, final_usage)
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.error("Stream error: %s", exc, exc_info=True)
                if not isinstance(exc, UsageLimitExceeded):
                    record_rotator_event("error")
                runtime_state.record(model, "errors")
                yield sse_error(str(exc))
            finally:
                cleanup_files(cleanup_paths)

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _remove_google_search_tool_names(tool_names: list[str]) -> list[str]:
    search_tools = {"google_search", "google_search_and_image_search", "image_search"}
    return [name for name in tool_names if name not in search_tools]


def _is_context_summarization_request(capture_prompt: str) -> bool:
    """Detect client-side context-compression prompts that should stay tool-free."""

    prompt = capture_prompt.strip().lower()
    return "summarization agent" in prompt and "creating a context" in prompt


async def _maybe_continue_incomplete_final_text(
    *,
    client: AIStudioClient,
    model: str,
    capture_prompt: str,
    capture_images: list[str] | None,
    contents: list[AistudioContent],
    system_instruction: str | None,
    partial_text: str,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    max_tokens: int | None,
    safety_settings: list[list] | None,
    generation_config_overrides: dict | None,
) -> tuple[str, bool]:
    """Try bounded text-only continuations when the final answer is visibly cut off.

    Some upstream model responses return finish_reason=stop after producing an
    obviously incomplete structured answer, most commonly a Markdown table after
    a successful tool result. This is not a tool-call continuation; it is a
    narrow text repair. Do not forward client tools on the repair request so the
    agent does not re-enter a tool loop.
    """

    current_text = partial_text
    continuations: list[str] = []
    max_repair_attempts = max(1, min(settings.openai_repair_max_attempts, 10))

    for repair_attempt in range(1, max_repair_attempts + 1):
        reason = _detect_incomplete_final_text(current_text)
        if not reason:
            break

        logger.warning(
            "OpenAI stream incomplete final text detected: model=%s, reason=%s, chars=%d, repair_attempt=%d/%d; requesting continuation",
            model,
            reason,
            len(current_text),
            repair_attempt,
            max_repair_attempts,
        )

        continuation_contents = _compact_continuation_contents(
            contents=contents,
            partial_text=current_text,
            reason=reason,
        )
        chunks: list[str] = []
        event_counts: dict[str, int] = {}
        try:
            async for event_type, text in client.stream_generate_content(
                model=model,
                capture_prompt=capture_prompt,
                capture_images=capture_images,
                contents=continuation_contents,
                system_instruction_content=None,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                tools=None,
                safety_settings=safety_settings,
                generation_config_overrides=_continuation_generation_config_overrides(generation_config_overrides),
                force_refresh_capture=False,
            ):
                event_counts[str(event_type)] = event_counts.get(str(event_type), 0) + 1
                if event_type == "body" and text:
                    chunks.append(str(text))
        except Exception as exc:
            logger.warning("OpenAI stream incomplete final text continuation failed: model=%s, error=%s", model, exc)
            break

        continuation_text = "".join(chunks)
        if not continuation_text:
            logger.warning(
                "OpenAI stream incomplete final text continuation produced no body: model=%s, reason=%s, events=%s; retrying non-stream",
                model,
                reason,
                event_counts,
            )
            try:
                output = await client.generate_content(
                    model=model,
                    capture_prompt=capture_prompt,
                    capture_images=capture_images,
                    contents=continuation_contents,
                    system_instruction_content=None,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_tokens,
                    tools=None,
                    safety_settings=safety_settings,
                    generation_config_overrides=_continuation_generation_config_overrides(generation_config_overrides),
                    sanitize_plain_text=True,
                )
                continuation_text = output.text or ""
            except Exception as exc:
                logger.warning(
                    "OpenAI stream incomplete final text non-stream continuation failed: model=%s, error=%s",
                    model,
                    exc,
                )
                break

        if not continuation_text:
            logger.warning(
                "OpenAI stream incomplete final text continuation still empty: model=%s, reason=%s, repair_attempt=%d/%d; trying next repair attempt",
                model,
                reason,
                repair_attempt,
                max_repair_attempts,
            )
            continue

        continuation_text = _join_repair_continuation(current_text, continuation_text)
        continuations.append(continuation_text)
        current_text += continuation_text
        logger.info(
            "OpenAI stream incomplete final text continued: model=%s, reason=%s, chars=%d, repair_attempt=%d/%d",
            model,
            reason,
            len(continuation_text),
            repair_attempt,
            max_repair_attempts,
        )

    final_reason = _detect_incomplete_final_text(current_text)
    if final_reason:
        logger.warning(
            "OpenAI stream incomplete final text remains incomplete after repair: model=%s, reason=%s, chars=%d, repairs=%d",
            model,
            final_reason,
            len(current_text),
            len(continuations),
        )
        logger.warning(
            "OpenAI stream suppressing unrepaired incomplete final text: model=%s, reason=%s, chars=%d",
            model,
            final_reason,
            len(current_text),
        )
        if _is_incomplete_pseudo_tool_call_reason(final_reason):
            return _unrepaired_pseudo_tool_call_notice(), True
        return _unrepaired_incomplete_final_text_notice(), True
    return "".join(continuations), False


def _is_incomplete_pseudo_tool_call_reason(reason: str) -> bool:
    return reason in {"incomplete_pseudo_tool_call_tag", "unclosed_pseudo_tool_call"}


def _unrepaired_pseudo_tool_call_notice() -> str:
    return (
        "模型生成了一个未完整的工具调用，系统已阻止将半截工具参数作为普通文本输出。"
        "请重新发送“继续”，或让任务从最近一步重新执行。"
    )


def _unrepaired_incomplete_final_text_notice() -> str:
    return (
        "模型生成的最终预览未能完整完成，系统已阻止将半截 Markdown/结构化内容继续输出。"
        "如果任务已经通过工具写入了文件，请以文件内容为准；也可以重新发送“继续”让任务从最近一步恢复。"
    )


def _join_repair_continuation(current_text: str, continuation_text: str) -> str:
    """Join repair text without accidentally merging Markdown table rows."""

    if (
        continuation_text
        and not current_text.endswith(("\n", "\r"))
        and _ends_with_complete_markdown_table_row(current_text)
        and continuation_text.lstrip().startswith("|")
    ):
        return "\n" + continuation_text.lstrip("\r\n")
    return continuation_text


def _continuation_prompt(reason: str) -> str:
    if _is_incomplete_pseudo_tool_call_reason(reason):
        return (
            "上一个回答在文本形式的工具调用标签中途停止了。"
            f"检测原因：{reason}。"
            "请只补全这个工具调用标签和参数，使其成为完整可解析的格式。"
            "不要解释，不要重复已经输出的前缀，不要输出其他正文。"
        )

    shortfall = _parse_markdown_table_shortfall_reason(reason)
    if shortfall is not None:
        expected, actual = shortfall
        missing = max(expected - actual, 1)
        return (
            "上一个回答承诺输出 Markdown 表格条目，但实际表格行数不足。"
            f"检测原因：{reason}。"
            f"请根据最近一次工具结果，只补齐缺少的剩余 {missing} 条，"
            f"不要重复已经输出的 {actual} 条，继续使用同一张 Markdown 表格格式。"
            "不要调用工具，不要重新生成开头，不要输出解释。"
        )

    return (
        "上一个回答在结构化输出中途停止了。"
        f"检测原因：{reason}。"
        "请从中断处继续完成剩余内容，不要重复已经输出的内容，"
        "不要调用工具，不要重新生成开头。"
    )


def _compact_continuation_contents(
    *,
    contents: list[AistudioContent],
    partial_text: str,
    reason: str,
) -> list[AistudioContent]:
    """Build a compact repair prompt instead of replaying the full conversation."""

    context_text = _latest_repair_context_text(contents)
    sections: list[str] = []
    if context_text:
        sections.append(
            "最近一次工具结果或上下文如下，仅用于补全上一个中断输出：\n"
            + _trim_repair_context(context_text)
        )
    sections.append("上一个回答已经输出到这里：\n" + partial_text)
    sections.append(_continuation_prompt(reason))
    return [AistudioContent(role="user", parts=[AistudioPart(text="\n\n".join(sections))])]


def _latest_repair_context_text(contents: list[AistudioContent]) -> str:
    """Return the latest useful textual context for a final-answer repair."""

    fallback = ""
    for content in reversed(contents):
        text_parts = [part.text for part in content.parts if getattr(part, "text", None)]
        if not text_parts:
            continue
        text = "\n".join(text_parts).strip()
        if not text:
            continue
        if "<tool_result" in text or '"snapshot"' in text or "snapshot:" in text:
            return text
        if not fallback:
            fallback = text
    return fallback


def _trim_repair_context(text: str, limit: int = 60000) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n\n[... repair context truncated ...]\n\n" + text[-tail:]


def _continuation_generation_config_overrides(generation_config_overrides: dict | None) -> dict[str, object | None]:
    """Use direct visible-text generation for repair requests.

    Repair prompts should produce a short visible continuation instead of spending
    the response budget on hidden reasoning.  Do not inherit a global high
    thinking default here.
    """

    overrides: dict[str, object | None] = dict(generation_config_overrides or {})
    overrides["thinking_config"] = AistudioThinkingConfig(
        level=ThinkingLevel.MINIMAL,
        mode=1,
    ).to_wire()
    return overrides


def _detect_incomplete_final_text(text: str) -> str | None:
    """Return a reason if final visible text appears obviously incomplete."""

    stripped = text.rstrip()
    if not stripped:
        return None

    pseudo_reason = _detect_incomplete_pseudo_tool_call(stripped)
    if pseudo_reason:
        return pseudo_reason

    if stripped.count("```") % 2 == 1:
        return "unclosed_code_fence"

    table_reason = _detect_incomplete_markdown_table(stripped)
    if table_reason:
        return table_reason

    short_table_reason = _detect_promised_markdown_table_shortfall(stripped)
    if short_table_reason:
        return short_table_reason

    if _ends_with_complete_markdown_table_row(stripped):
        return None

    if stripped.endswith(("|", ",", "{", "[", "(", "（", "、", "，", ":", "：")):
        return "dangling_terminal"

    if _looks_like_unclosed_json(stripped):
        return "unclosed_json"

    return None


def _detect_incomplete_pseudo_tool_call(text: str) -> str | None:
    """Detect model-visible pseudo tool markup that stopped mid-tag.

    This is a narrow guard for cases where the model starts emitting a textual
    tool invocation but the upstream model service returns finish_reason=stop
    before the tag is syntactically complete, leaving OpenAI-compatible clients
    with plain text instead of an executable tool call.
    """

    tag_name = "<" + "tool_call"
    close_name = "</" + "tool_call" + ">"
    last_open = text.rfind(tag_name)
    if last_open == -1:
        return None
    last_close = text.rfind(close_name)
    if last_close > last_open:
        return None

    fragment = text[last_open:].strip()
    if not fragment:
        return None

    if ">" not in fragment:
        return "incomplete_pseudo_tool_call_tag"
    if close_name not in fragment:
        return "unclosed_pseudo_tool_call"

    return None


def _detect_incomplete_markdown_table(text: str) -> str | None:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None

    table_indexes = [idx for idx, line in enumerate(lines) if line.lstrip().startswith("|")]
    if len(table_indexes) < 2:
        return None

    last_line = lines[-1]
    if last_line.lstrip().startswith("|"):
        header_idx = table_indexes[0]
        header_cols = _markdown_table_column_count(lines[header_idx])
        last_cols = _markdown_table_column_count(last_line)
        if header_cols >= 2 and last_cols < header_cols:
            return "incomplete_markdown_table_row"
        if not last_line.rstrip().endswith("|"):
            return "unterminated_markdown_table_row"

    if len(table_indexes) >= 2 and lines[table_indexes[-1]].strip() in {"|", "||"}:
        return "empty_markdown_table_row"

    return None


def _detect_promised_markdown_table_shortfall(text: str) -> str | None:
    expected = _promised_table_row_count(text)
    if expected is None or expected <= 0:
        return None

    actual = _markdown_table_data_row_count(text)
    if actual <= 0 and not _has_markdown_table_header(text):
        return None
    if actual < expected:
        return f"short_markdown_table_rows:{expected}:{actual}"
    return None


def _promised_table_row_count(text: str) -> int | None:
    candidates: list[int] = []
    for match in re.finditer(r"(\d{1,4})\s*[-~～—－]\s*(\d{1,4})\s*条", text):
        try:
            start = int(match.group(1))
            end = int(match.group(2))
        except ValueError:
            continue
        if start <= end:
            count = end - start + 1
            if 2 <= count <= 100:
                candidates.append(count)

    patterns = (
        r"(?:继续显示|显示|列出|输出|整理|返回|提供|前|后|最近)\s*(\d{1,3})\s*条",
        r"(\d{1,3})\s*条(?:结果|记录|帖子|条目)",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if 2 <= value <= 100:
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates)


def _has_markdown_table_header(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return False
    return any(_is_markdown_table_separator(line) for line in lines[1:])


def _markdown_table_data_row_count(text: str) -> int:
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return 0

    data_rows = 0
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if _is_markdown_table_separator(line):
            continue
        if _markdown_table_column_count(line) >= 2:
            data_rows += 1
    return data_rows


def _is_markdown_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(cell and set(cell) <= {"-", ":"} for cell in cells)


def _parse_markdown_table_shortfall_reason(reason: str) -> tuple[int, int] | None:
    prefix = "short_markdown_table_rows:"
    if not reason.startswith(prefix):
        return None
    parts = reason[len(prefix) :].split(":")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _markdown_table_column_count(line: str) -> int:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    if not stripped:
        return 0
    return len(stripped.split("|"))


def _ends_with_complete_markdown_table_row(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.lstrip().startswith("|")]
    if len(table_lines) < 2:
        return False
    last_line = lines[-1]
    if not last_line.lstrip().startswith("|") or not last_line.rstrip().endswith("|"):
        return False

    header_cols = _markdown_table_column_count(table_lines[0])
    last_cols = _markdown_table_column_count(last_line)
    return header_cols >= 2 and last_cols >= header_cols


def _looks_like_unclosed_json(text: str) -> bool:
    first = text.lstrip()[:1]
    if first not in {"{", "["}:
        return False
    try:
        json.loads(text)
        return False
    except json.JSONDecodeError:
        return True


def _request_field_set(req: ChatRequest, field: str) -> bool:
    """Return whether a Pydantic request explicitly included a field."""

    fields_set = getattr(req, "model_fields_set", set())
    return field in fields_set


def _resolve_openai_request_options(req: ChatRequest) -> dict[str, object]:
    """Apply .env defaults to OpenAI-compatible chat requests.

    Request values take precedence over AISTUDIO_OPENAI_DEFAULT_* settings.  For
    boolean extension fields, Pydantic's model_fields_set lets clients explicitly
    send false to override a true environment default.
    """

    thinking = req.thinking if _request_field_set(req, "thinking") else settings.openai_default_thinking
    safety_off = req.safety_off if _request_field_set(req, "safety_off") else settings.openai_default_safety_off
    google_search = (
        req.google_search if _request_field_set(req, "google_search") else settings.openai_default_google_search
    )

    generation_config_overrides: dict[str, object | None] = {}
    if thinking is not None:
        generation_config_overrides["thinking_config"] = _openai_thinking_config(thinking)

    return {
        "temperature": req.temperature if req.temperature is not None else settings.openai_default_temperature,
        "top_p": req.top_p if req.top_p is not None else settings.openai_default_top_p,
        "top_k": req.top_k if req.top_k is not None else settings.openai_default_top_k,
        "max_tokens": req.max_tokens if req.max_tokens is not None else settings.openai_default_max_tokens,
        "google_search": google_search,
        "safety_settings": _safety_off_settings() if safety_off else None,
        "generation_config_overrides": generation_config_overrides or None,
    }


def _openai_thinking_config(value: str):
    label = str(value).strip().lower()
    if label in {"", "off", "none", "false", "0"}:
        return None
    level_map = {
        "low": ThinkingLevel.LOW,
        "medium": ThinkingLevel.MEDIUM,
        "mid": ThinkingLevel.MEDIUM,
        "high": ThinkingLevel.HIGH,
        "minimal": ThinkingLevel.MINIMAL,
        "min": ThinkingLevel.MINIMAL,
    }
    if label not in level_map:
        raise ValueError(f"Unsupported OpenAI thinking default: {value!r}")
    return AistudioThinkingConfig(level=level_map[label], mode=1).to_wire()


def _safety_off_settings() -> list[list]:
    return [[None, None, cat, 5] for cat in [7, 8, 9, 10]]


def _last_non_system_role(messages) -> str:
    for msg in reversed(messages):
        role = (msg.role or "").lower()
        if role not in {"system", "developer"}:
            return role
    return ""


_PSEUDO_TOOL_CALL_RE = re.compile(
    r"<tool_call\s+([^>]*)>\s*(.*?)\s*</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_PSEUDO_TOOL_ATTR_RE = re.compile(r"(\w+)\s*=\s*(['\"])(.*?)\2", re.DOTALL)


def _extract_pseudo_tool_calls(text: str) -> list[dict]:
    """Convert textual <tool_call ...>{...}</tool_call> blocks to function calls.

    Some upstream models occasionally emit the agent tool-call transcript as plain
    text instead of native function calls.  OpenAI-compatible agent clients only
    continue when the response contains real tool_calls, so translate complete
    transcript blocks back into function calls as a compatibility fallback.
    """

    calls: list[dict] = []
    for match in _PSEUDO_TOOL_CALL_RE.finditer(text):
        attrs = {key: value for key, _quote, value in _PSEUDO_TOOL_ATTR_RE.findall(match.group(1))}
        name = attrs.get("name") or attrs.get("function")
        if not name:
            continue

        raw_args = match.group(2).strip()
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            args = {"input": raw_args}
        if not isinstance(args, dict):
            args = {"input": args}

        call: dict = {"name": name, "args": args}
        call_id = attrs.get("tool_call_id") or attrs.get("id") or attrs.get("call_id")
        if call_id:
            call["call_id"] = call_id
        calls.append(call)

    if calls:
        return calls

    # Some models emit a malformed, attribute-only transcript such as:
    #   <tool_call name="execute_code" code="..."}
    # instead of a complete <tool_call>JSON</tool_call> block.  Recover it so
    # agent clients still receive a native OpenAI tool_call rather than plain
    # text.
    for attrs in _extract_pseudo_tool_call_attrs(text):
        name = attrs.pop("name", None) or attrs.pop("function", None)
        if not name:
            continue

        call_id = attrs.pop("tool_call_id", None) or attrs.pop("id", None) or attrs.pop("call_id", None)
        if not attrs:
            continue

        call = {"name": name, "args": attrs}
        if call_id:
            call["call_id"] = call_id
        calls.append(call)

    return calls


def _extract_pseudo_tool_call_attrs(text: str) -> list[dict[str, str]]:
    attrs_list: list[dict[str, str]] = []
    pos = 0
    marker = "<tool_call"

    while True:
        start = text.lower().find(marker, pos)
        if start < 0:
            break

        i = start + len(marker)
        attrs: dict[str, str] = {}
        while i < len(text):
            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] == ">" or text[i] == "<":
                break

            key_match = re.match(r"[A-Za-z_][A-Za-z0-9_\-]*", text[i:])
            if not key_match:
                break
            key = key_match.group(0)
            i += len(key)

            while i < len(text) and text[i].isspace():
                i += 1
            if i >= len(text) or text[i] != "=":
                break
            i += 1
            while i < len(text) and text[i].isspace():
                i += 1

            if i >= len(text) or text[i] not in {'"', "'"}:
                break
            quote = text[i]
            i += 1
            if key == "code":
                # Code snippets can contain many literal quotes.  In malformed
                # attribute-only transcripts, the code attribute is effectively
                # the final payload, so read greedily up to the last closing
                # quote before the optional malformed JSON tail.
                tail = text[i:]
                greedy_match = re.match(rf"(.*){re.escape(quote)}\s*\}}?\s*\]?\s*$", tail, re.DOTALL)
                if greedy_match:
                    raw_value = greedy_match.group(1)
                    raw_value = raw_value.replace(r"\"", '"').replace(r"\'", "'")
                    attrs[key] = _clean_pseudo_tool_attr_value(key, raw_value)
                    i = len(text)
                    continue

            escaped = False
            value_chars: list[str] = []
            while i < len(text):
                ch = text[i]
                if escaped:
                    value_chars.append(ch)
                    escaped = False
                elif ch == "\\":
                    value_chars.append(ch)
                    escaped = True
                elif ch == quote:
                    break
                else:
                    value_chars.append(ch)
                i += 1
            if i >= len(text) or text[i] != quote:
                break
            raw_value = "".join(value_chars)
            raw_value = raw_value.replace(r"\"", '"').replace(r"\'", "'")
            attrs[key] = _clean_pseudo_tool_attr_value(key, raw_value)
            i += 1

        if attrs:
            attrs_list.append(attrs)
        pos = max(i + 1, start + len(marker))

    return attrs_list


def _clean_pseudo_tool_attr_value(key: str, value: str) -> str:
    if key != "code":
        return value

    lines = value.splitlines()
    while lines and lines[-1].strip() == "}":
        lines.pop()
    if not lines:
        return ""
    return "\n".join(lines) + ("\n" if value.endswith("\n") else "")
