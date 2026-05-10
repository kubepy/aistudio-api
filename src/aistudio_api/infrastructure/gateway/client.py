"""Browser-backed AI Studio client facade."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from aistudio_api.config import DEFAULT_CAMOUFOX_PORT, DEFAULT_IMAGE_MODEL, DEFAULT_TEXT_MODEL, settings
from aistudio_api.domain.errors import RequestError, classify_error
from aistudio_api.domain.models import ModelOutput, parse_image_output, parse_text_output
from aistudio_api.infrastructure.cache.snapshot_cache import SnapshotCache
from aistudio_api.infrastructure.gateway.capture import CapturedRequest, RequestCaptureService
from aistudio_api.infrastructure.gateway.request_rewriter import TOOLS_TEMPLATES, modify_body
from aistudio_api.infrastructure.gateway.replay import RequestReplayService
from aistudio_api.infrastructure.gateway.session import BrowserSession
from aistudio_api.infrastructure.gateway.streaming import StreamingGateway
from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart

logger = logging.getLogger("aistudio")

_snapshot_cache = SnapshotCache()


class AIStudioClient:
    def __init__(self, port: int = DEFAULT_CAMOUFOX_PORT, use_pure_http: bool = False):
        self.port = port
        self._use_pure_http = use_pure_http
        self._captured: Optional[CapturedRequest] = None
        
        if use_pure_http:
            # Pure HTTP mode: no browser needed for capture
            from aistudio_api.infrastructure.gateway.pure_capture import PureHttpCaptureService
            self._capture_service = PureHttpCaptureService(_snapshot_cache)
            self._session = None
            self._replay_service = RequestReplayService(session=None)
        else:
            # Browser mode: uses browser for capture and replay
            self._session = BrowserSession(port=port)
            self._capture_service = RequestCaptureService(self._session, _snapshot_cache)
            self._replay_service = RequestReplayService(session=self._session)
        
        self._streaming_gateway = StreamingGateway(session=self._session)

    async def warmup(self) -> None:
        """预热浏览器，启动 Camoufox 并加载 AI Studio 页面。"""
        if self._session is not None:
            await self._session.ensure_context()
            logger.info("浏览器预热完成")

    async def switch_auth(self, auth_file: str | None) -> None:
        """切换账号的 auth 文件。"""
        if self._session is not None:
            await self._session.switch_auth(auth_file)

    def clear_snapshot_cache(self) -> None:
        """清除 snapshot 缓存。"""
        _snapshot_cache.clear()

    def _dump_raw_exchange(
        self,
        *,
        kind: str,
        model: str,
        capture_prompt: str,
        modified_body: str,
        raw_response: str,
    ) -> None:
        if not settings.dump_raw_response:
            return

        out_dir = Path(settings.dump_raw_response_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        safe_model = model.replace("/", "_")
        timestamp = __import__("time").strftime("%Y%m%d_%H%M%S")
        payload = {
            "kind": kind,
            "model": model,
            "capture_prompt": capture_prompt,
            "modified_body": json.loads(modified_body),
            "raw_response": raw_response,
        }
        path = out_dir / f"aistudio_{kind}_{safe_model}_{timestamp}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("已落盘原始请求/响应: %s", path)

    async def capture_request(
        self,
        prompt: str,
        model: str = DEFAULT_TEXT_MODEL,
        images: Optional[list[str]] = None,
        contents: Optional[list[AistudioContent]] = None,
        force_refresh: bool = False,
    ) -> Optional[CapturedRequest]:
        return await self._capture_service.capture(
            prompt=prompt,
            model=model,
            images=images,
            contents=contents,
            force_refresh=force_refresh,
        )

    async def replay(self, body: str, timeout: int = 120) -> tuple[int, bytes]:
        return await self._replay_service.replay(self._captured, body=body, timeout=timeout)

    async def stream_chat(
        self,
        *,
        prompt: str,
        model: str = DEFAULT_TEXT_MODEL,
        images: Optional[list[str]] = None,
        system_instruction: str | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: list[list] | None = None,
    ):
        merged_tools = list(tools or [])
        async for event in self.stream_generate_content(
            model=model,
            capture_prompt=prompt,
            capture_images=images,
            contents=[self._build_user_content(prompt=prompt, images=images)],
            system_instruction_content=(
                AistudioContent(role="user", parts=[AistudioPart(text=system_instruction)])
                if system_instruction
                else None
            ),
            tools=merged_tools or None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        ):
            yield event

    async def stream_generate_content(
        self,
        *,
        model: str = DEFAULT_TEXT_MODEL,
        capture_prompt: str,
        capture_images: Optional[list[str]] = None,
        contents: Optional[list[AistudioContent]] = None,
        system_instruction_content: AistudioContent | None = None,
        tools: list[list] | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        generation_config_overrides: dict | None = None,
        sanitize_plain_text: bool = True,
        force_refresh_capture: bool = False,
    ):
        captured = await self.capture_request(
            prompt=capture_prompt,
            model=model,
            images=capture_images,
            contents=contents,
            force_refresh=force_refresh_capture,
        )
        async for event in self._streaming_gateway.stream_chat(
            captured=captured,
            model=model,
            system_instruction=None,
            contents=contents,
            system_instruction_content=system_instruction_content,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            generation_config_overrides=generation_config_overrides,
            sanitize_plain_text=sanitize_plain_text,
        ):
            yield event

    async def chat(
        self,
        prompt: str,
        model: str = DEFAULT_TEXT_MODEL,
        system_instruction: Optional[str] = None,
        code_execution: bool = False,
        google_search: bool = False,
        images: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: list[list] | None = None,
    ) -> ModelOutput:
        merged_tools = list(tools or [])
        if code_execution or google_search:
            if code_execution:
                merged_tools.append(TOOLS_TEMPLATES["code_execution"])
            if google_search:
                merged_tools.append(TOOLS_TEMPLATES["google_search"])

        return await self.generate_content(
            model=model,
            capture_prompt=prompt,
            capture_images=images,
            contents=[self._build_user_content(prompt=prompt, images=images)],
            system_instruction_content=(
                AistudioContent(role="user", parts=[AistudioPart(text=system_instruction)])
                if system_instruction
                else None
            ),
            tools=merged_tools or None,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
        )

    async def generate_content(
        self,
        *,
        model: str = DEFAULT_TEXT_MODEL,
        capture_prompt: str,
        capture_images: Optional[list[str]] = None,
        contents: Optional[list[AistudioContent]] = None,
        system_instruction_content: AistudioContent | None = None,
        tools: list[list] | None = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        generation_config_overrides: dict | None = None,
        sanitize_plain_text: bool = True,
    ) -> ModelOutput:
        logger.info("拦截请求: %r", f"{capture_prompt[:20]}...")
        captured = await self.capture_request(capture_prompt, model=model, images=capture_images, contents=contents)
        if not captured:
            raise RequestError(0, "无法拦截请求")

        modified_body = modify_body(
            captured.body,
            model=model,
            contents=contents,
            system_instruction_content=system_instruction_content,
            tools=tools,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            generation_config_overrides=generation_config_overrides,
            sanitize_plain_text=sanitize_plain_text,
        )

        status, raw = await self._replay_service.replay(captured, body=modified_body)
        raw_text = raw.decode("utf-8", errors="replace")
        self._dump_raw_exchange(
            kind="generate_content",
            model=model,
            capture_prompt=capture_prompt,
            modified_body=modified_body,
            raw_response=raw_text,
        )
        if status != 200:
            raise classify_error(status, raw_text)
        output = parse_text_output(raw_text)
        output.model = model
        return output

    @staticmethod
    def _convert_size_to_resolution(size: str) -> list:
        """将 OpenAI 格式的 size 转换为 AI Studio 格式的 resolution."""
        size_map = {
            # 1:1
            "512x512": ["1:1", "512"],
            "1024x1024": ["1:1", "1K"],
            "2048x2048": ["1:1", "2K"],
            "4096x4096": ["1:1", "4K"],
            # 16:9
            "1792x1024": ["16:9", "1K"],
            # 9:16
            "1024x1792": ["9:16", "1K"],
            # 4:3
            "1365x1024": ["4:3", "1K"],
            # 3:4
            "1024x1365": ["3:4", "1K"],
            # 3:2
            "1536x1024": ["3:2", "1K"],
            # 2:3
            "1024x1536": ["2:3", "1K"],
        }
        return size_map.get(size, ["1:1", "1K"])

    async def generate_image(
        self,
        prompt: str,
        model: str = DEFAULT_IMAGE_MODEL,
        save_path: Optional[str] = None,
        size: str = "1024x1024",
        google_search: bool = False,
    ) -> ModelOutput:
        logger.info("生图请求: %r", f"{prompt[:20]}...")
        captured = await self.capture_request(prompt, model=model)
        if not captured:
            raise RequestError(0, "无法拦截请求")

        modified_body = modify_body(captured.body, model=model, prompt=prompt)
        status, raw = await self._replay_service.replay(captured, body=modified_body, timeout=120)
        raw_text = raw.decode("utf-8", errors="replace")
        self._dump_raw_exchange(
            kind="generate_image",
            model=model,
            capture_prompt=prompt,
            modified_body=modified_body,
            raw_response=raw_text,
        )
        if status != 200:
            raise classify_error(status, raw_text)
        output = parse_image_output(raw_text)
        output.model = model

        if output.images:
            img = output.images[0]
            ext = "jpg" if "jpeg" in img.mime else "png"
            path = save_path if save_path and save_path.endswith(f".{ext}") else (
                f"{save_path}.{ext}" if save_path else f"/tmp/aistudio_generated.{ext}"
            )
            with open(path, "wb") as file:
                file.write(img.data)
            logger.info("图片已保存: %s (%s bytes)", path, img.size)

        return output

    def _build_user_content(self, prompt: str, images: Optional[list[str]] = None) -> AistudioContent:
        import base64
        import mimetypes

        parts = []
        for image_path in images or []:
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as file:
                parts.append(AistudioPart(inline_data=(mime, base64.b64encode(file.read()).decode("ascii"))))
        parts.append(AistudioPart(text=prompt))
        return AistudioContent(role="user", parts=parts)


from aistudio_api.infrastructure.gateway.cli import cli_main

__all__ = ["AIStudioClient", "CapturedRequest", "cli_main"]
