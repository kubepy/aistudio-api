import asyncio

import pytest
from fastapi import HTTPException

from aistudio_api.api.schemas import ImageRequest
from aistudio_api.application.api_service import handle_image_edit, handle_image_generation


class _UnusedClient:
    async def generate_image(self, *args, **kwargs):
        raise AssertionError("generate_image should not be called for invalid requests")


def test_handle_image_generation_rejects_unsupported_n():
    req = ImageRequest(prompt="hello", n=2)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(handle_image_generation(req, _UnusedClient()))

    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "invalid_request_error"
    assert exc.value.detail["message"] == "Only n=1 is currently supported"


def test_handle_image_edit_rejects_unsupported_size():
    with pytest.raises(HTTPException) as exc:
        asyncio.run(
            handle_image_edit(
                prompt="hello",
                image_files=[],
                mask_file=None,
                model="gemini-3.1-flash-image-preview",
                n=1,
                size="800x600",
                temperature=None,
                top_p=None,
                client=_UnusedClient(),
            )
        )

    assert exc.value.status_code == 400
    assert exc.value.detail["type"] == "invalid_request_error"
    assert exc.value.detail["message"] == "Unsupported image size '800x600'"


def test_native_aistudio_aspect_ratio_aliases_are_supported():
    from aistudio_api.application.api_service_common import validate_image_request_options
    from aistudio_api.infrastructure.gateway.client import AIStudioClient

    expected_aliases = {
        "1:1": ["1:1", "1K"],
        "9:16": ["9:16", "1K"],
        "16:9": ["16:9", "1K"],
        "3:4": ["3:4", "1K"],
        "4:3": ["4:3", "1K"],
        "3:2": ["3:2", "1K"],
        "2:3": ["2:3", "1K"],
        "5:4": ["5:4", "1K"],
        "4:5": ["4:5", "1K"],
        "21:9": ["21:9", "1K"],
    }

    for size, expected in expected_aliases.items():
        validate_image_request_options(size=size, n=1)
        assert AIStudioClient.resolve_image_size(size) == expected


class _CaptureImageClient:
    def __init__(self):
        self.calls = []

    async def generate_image(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})

        class _Output:
            images = []
            text = ""
            usage = None

        return _Output()


def test_handle_image_generation_passes_split_search_flags(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureImageClient()
    req = ImageRequest(prompt="hello", google_search=True, image_search=True)

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["google_search"] is True
    assert client.calls[0]["kwargs"]["image_search"] is True
    assert client.calls[0]["kwargs"]["use_default_tools"] is False


def test_handle_image_generation_allows_default_tools_when_search_flags_omitted(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureImageClient()
    req = ImageRequest(prompt="hello")

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["use_default_tools"] is True

def test_handle_image_generation_passes_sampling_options(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureImageClient()
    req = ImageRequest(prompt="hello", temperature=0.7, top_p=0.9)

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["temperature"] == 0.7
    assert client.calls[0]["kwargs"]["top_p"] == 0.9

def test_handle_image_generation_force_refreshes_after_account_switch(monkeypatch):
    from aistudio_api.application import api_service_openai
    from aistudio_api.domain.errors import UsageLimitExceeded

    async def _noop(*args, **kwargs):
        return None

    switch_calls = []

    async def _switch():
        switch_calls.append(True)
        return True

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_openai, "try_switch_account", _switch)

    class _Client:
        def __init__(self):
            self.calls = []

        async def generate_image(self, *args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise UsageLimitExceeded("limited")

            class _Output:
                images = []
                text = ""
                usage = None

            return _Output()

    client = _Client()
    req = ImageRequest(prompt="hello")

    asyncio.run(handle_image_generation(req, client))

    assert len(switch_calls) == 1
    assert len(client.calls) == 2
    assert client.calls[0]["force_refresh_capture"] is False
    assert client.calls[1]["force_refresh_capture"] is True


def test_handle_image_generation_retries_capture_replay_error_with_force_refresh(monkeypatch):
    from aistudio_api.application import api_service_openai
    from aistudio_api.domain.errors import RequestError

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    class _Client:
        def __init__(self):
            self.calls = []

        async def generate_image(self, *args, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise RequestError(0, "no captured URL available for replay")

            class _Output:
                images = []
                text = ""
                usage = None

            return _Output()

    client = _Client()
    req = ImageRequest(prompt="hello")

    asyncio.run(handle_image_generation(req, client))

    assert len(client.calls) == 2
    assert client.calls[0]["force_refresh_capture"] is False
    assert client.calls[1]["force_refresh_capture"] is True

def test_try_switch_account_clears_capture_cache(monkeypatch):
    from aistudio_api.application import api_service_common
    from aistudio_api.api.state import runtime_state

    class _Account:
        id = "acc_next"

    class _Rotator:
        async def get_next_account(self):
            return _Account()

    class _AccountService:
        async def activate_account(self, account_id, session, snapshot_cache, busy_lock, keep_snapshot_cache):
            assert account_id == "acc_next"
            assert busy_lock is None
            assert keep_snapshot_cache is False
            return _Account()

    class _Client:
        def __init__(self):
            self._session = object()
            self.clear_calls = 0

        def clear_capture_cache(self):
            self.clear_calls += 1

    client = _Client()
    original = (
        runtime_state.rotator,
        runtime_state.account_service,
        runtime_state.client,
        runtime_state.snapshot_cache,
    )
    runtime_state.rotator = _Rotator()
    runtime_state.account_service = _AccountService()
    runtime_state.client = client
    runtime_state.snapshot_cache = object()
    try:
        assert asyncio.run(api_service_common.try_switch_account()) is True
        assert client.clear_calls == 1
    finally:
        (
            runtime_state.rotator,
            runtime_state.account_service,
            runtime_state.client,
            runtime_state.snapshot_cache,
        ) = original
