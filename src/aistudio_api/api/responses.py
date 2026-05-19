"""OpenAI-compatible HTTP/SSE response helpers."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from pydantic import BaseModel

from aistudio_api.api.response_models import (
    AnthropicMessageResponse,
    AnthropicTextBlockResponse,
    AnthropicToolUseBlockResponse,
    AnthropicUsageResponse,
    ErrorDetail,
    ErrorResponse,
    GeminiFunctionCallPayload,
    GeminiFunctionResponsePayload,
    GeminiPartResponse,
    GeminiUsageMetadata,
    OpenAIChatChoice,
    OpenAIChatChunkChoice,
    OpenAIChatCompletionChunk,
    OpenAIChatCompletionResponse,
    OpenAIChatDelta,
    OpenAIChatMessage,
    OpenAIFunctionCallPayload,
    OpenAIToolCall,
    OpenAIUsage,
)


def new_chat_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def normalize_usage(usage: dict | None = None) -> OpenAIUsage:
    completion_details = (usage or {}).get("completion_tokens_details") or {}
    return OpenAIUsage(
        prompt_tokens=(usage or {}).get("prompt_tokens", 0) or 0,
        completion_tokens=(usage or {}).get("completion_tokens", 0) or 0,
        total_tokens=(usage or {}).get("total_tokens", 0) or 0,
        completion_tokens_details={"reasoning_tokens": completion_details.get("reasoning_tokens", 0) or 0},
    )


def to_gemini_usage_metadata(usage: dict | None = None) -> GeminiUsageMetadata:
    completion_details = (usage or {}).get("completion_tokens_details") or {}
    reasoning_tokens = completion_details.get("reasoning_tokens", 0) or 0
    visible_tokens = completion_details.get("visible_tokens")
    candidates_tokens = visible_tokens if visible_tokens is not None else (usage or {}).get("completion_tokens", 0)
    return GeminiUsageMetadata(
        promptTokenCount=(usage or {}).get("prompt_tokens", 0) or 0,
        candidatesTokenCount=candidates_tokens or 0,
        thoughtsTokenCount=reasoning_tokens,
        totalTokenCount=(usage or {}).get("total_tokens", 0) or 0,
    )


def sse_chunk(
    chat_id: str,
    model: str,
    content: str,
    finish: str | None = None,
    thinking: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    include_usage: bool = True,
) -> str:
    data = OpenAIChatCompletionChunk(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[
            OpenAIChatChunkChoice(
                index=0,
                delta=OpenAIChatDelta(
                    content=content or None,
                    thinking=thinking,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish,
            )
        ],
        usage=None,
    )
    payload = data.model_dump(mode="json", exclude_none=True)
    if include_usage:
        payload["usage"] = None
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def sse_usage_chunk(chat_id: str, model: str, usage: dict | None = None) -> str:
    data = OpenAIChatCompletionChunk(
        id=chat_id,
        created=int(time.time()),
        model=model,
        choices=[],
        usage=normalize_usage(usage),
    )
    return f"data: {data.model_dump_json()}\n\n"


def sse_error(message: str) -> str:
    data = ErrorResponse(error=ErrorDetail(message=message, type="server_error"))
    return f"data: {data.model_dump_json()}\n\n"


def anthropic_usage(usage: dict | None = None) -> AnthropicUsageResponse:
    normalized = normalize_usage(usage)
    return AnthropicUsageResponse(
        input_tokens=normalized.prompt_tokens,
        output_tokens=normalized.completion_tokens,
    )


def anthropic_sse(event_type: str, payload: dict[str, Any] | BaseModel) -> str:
    if isinstance(payload, BaseModel):
        body = payload.model_dump_json()
    else:
        body = json.dumps(payload, ensure_ascii=False)
    return f"event: {event_type}\ndata: {body}\n\n"


def anthropic_error_sse(message: str) -> str:
    return anthropic_sse(
        "error",
        {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        },
    )


def function_call_args(function_call: dict[str, Any]) -> dict[str, Any]:
    if isinstance(function_call.get("args"), dict):
        return function_call["args"]
    if "arguments" in function_call:
        raw_args = function_call["arguments"]
        if isinstance(raw_args, dict):
            return raw_args
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError:
                return {"arguments": raw_args}
            return parsed if isinstance(parsed, dict) else {"arguments": parsed}
    raw = function_call.get("raw")
    if isinstance(raw, list) and len(raw) > 1:
        raw_args = raw[1]
        return raw_args if isinstance(raw_args, dict) else {"arguments": raw_args}
    return {}


def anthropic_message_response(
    *,
    model: str,
    content: str,
    usage: dict | None = None,
    function_calls: list[dict[str, Any]] | None = None,
) -> AnthropicMessageResponse:
    blocks: list[AnthropicTextBlockResponse | AnthropicToolUseBlockResponse] = []
    if content:
        blocks.append(AnthropicTextBlockResponse(text=content))
    for function_call in function_calls or []:
        blocks.append(
            AnthropicToolUseBlockResponse(
                id=function_call.get("anthropic_tool_use_id") or f"toolu_{uuid.uuid4().hex[:24]}",
                name=function_call.get("name", "unknown"),
                input=function_call_args(function_call),
            )
        )
    if not blocks:
        blocks.append(AnthropicTextBlockResponse(text=""))

    return AnthropicMessageResponse(
        id=new_message_id(),
        model=model,
        content=blocks,
        stop_reason="tool_use" if function_calls else "end_turn",
        stop_sequence=None,
        usage=anthropic_usage(usage),
    )


def _function_call_arguments(function_call: dict[str, Any]) -> str:
    if "args" in function_call:
        return json.dumps(function_call["args"], ensure_ascii=False)
    if "arguments" in function_call:
        return str(function_call["arguments"])
    raw = function_call.get("raw")
    if isinstance(raw, list) and len(raw) > 1:
        second = raw[1]
        if isinstance(second, str):
            return second
        return json.dumps(second, ensure_ascii=False)
    return "{}"


def to_openai_tool_calls(function_calls: list[dict[str, Any]]) -> list[OpenAIToolCall]:
    tool_calls: list[OpenAIToolCall] = []
    for idx, function_call in enumerate(function_calls):
        tool_calls.append(
            OpenAIToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}_{idx}",
                function=OpenAIFunctionCallPayload(
                    name=function_call.get("name", "unknown"),
                    arguments=_function_call_arguments(function_call),
                ),
            )
        )
    return tool_calls


def to_gemini_parts(
    content: str,
    function_calls: list[dict[str, Any]] | None = None,
    function_responses: list[dict[str, Any]] | None = None,
    thinking: str = "",
) -> list[GeminiPartResponse]:
    parts: list[GeminiPartResponse] = []
    if thinking:
        parts.append(GeminiPartResponse(text=thinking, thought=True))
    if content:
        parts.append(GeminiPartResponse(text=content))
    for function_call in function_calls or []:
        payload = GeminiFunctionCallPayload(name=function_call.get("name", "unknown"))
        if "args" in function_call:
            payload.args = function_call["args"]
        elif "arguments" in function_call:
            payload.args = function_call["arguments"]
        elif isinstance(function_call.get("raw"), list) and len(function_call["raw"]) > 1:
            payload.args = function_call["raw"][1]
        parts.append(GeminiPartResponse(functionCall=payload))
    for function_response in function_responses or []:
        payload = GeminiFunctionResponsePayload(name=function_response.get("name", "unknown"))
        if "args" in function_response:
            payload.response = function_response["args"]
        elif "arguments" in function_response:
            payload.response = function_response["arguments"]
        elif isinstance(function_response.get("raw"), list) and len(function_response["raw"]) > 1:
            payload.response = function_response["raw"][1]
        parts.append(GeminiPartResponse(functionResponse=payload))
    if not parts:
        parts.append(GeminiPartResponse(text=""))
    return parts


def chat_completion_response(
    model: str,
    content: str,
    thinking: str = "",
    usage: dict | None = None,
    function_calls: list[dict[str, Any]] | None = None,
) -> OpenAIChatCompletionResponse:
    finish_reason = "tool_calls" if function_calls else "stop"
    return OpenAIChatCompletionResponse(
        id=new_chat_id(),
        created=int(time.time()),
        model=model,
        choices=[
            OpenAIChatChoice(
                index=0,
                message=OpenAIChatMessage(
                    content=content,
                    thinking=thinking or None,
                    tool_calls=to_openai_tool_calls(function_calls) if function_calls else None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=normalize_usage(usage),
    )
