import json
import asyncio

from aistudio_api.infrastructure.cache.snapshot_cache import SnapshotCache
from aistudio_api.infrastructure.gateway.capture import RequestCaptureService


class _FakeSession:
    async def capture_template(self, model: str, force_refresh: bool = False):
        # Simulate AI Studio UI capturing a template from the browser default model,
        # while the API request asks for a different model.
        return {
            "url": "https://example.test/StreamGenerateContent",
            "headers": {},
            "body": json.dumps(
                [
                    "models/gemma-4-31b-it",
                    [[[[None, "say t"]], "user"]],
                    [[None, None, 7, 4]],
                    [None, None, None, 8192, 1.0, 0.95],
                    "template-snapshot",
                    None,
                    None,
                    None,
                    None,
                    None,
                    1,
                ]
            ),
        }

    async def generate_snapshot(self, contents):
        return "request-snapshot"


def test_capture_uses_requested_model_not_template_model():
    service = RequestCaptureService(_FakeSession(), SnapshotCache())

    captured = asyncio.run(service.capture(prompt="hello", model="models/gemini-3.5-flash"))

    body = json.loads(captured.body)
    assert body[0] == "models/gemini-3.5-flash"
    assert body[4] == "request-snapshot"

def test_capture_force_refresh_recaptures_template():
    class _ChangingSession(_FakeSession):
        def __init__(self):
            self.template_calls = 0

        async def capture_template(self, model: str, force_refresh: bool = False):
            self.template_calls += 1
            captured = await super().capture_template(model, force_refresh=force_refresh)
            captured["url"] = f"https://example.test/StreamGenerateContent/{self.template_calls}"
            return captured

    session = _ChangingSession()
    service = RequestCaptureService(session, SnapshotCache())

    first = asyncio.run(service.capture(prompt="hello", model="models/gemini-3.5-flash"))
    cached = asyncio.run(service.capture(prompt="hello", model="models/gemini-3.5-flash"))
    refreshed = asyncio.run(
        service.capture(prompt="hello", model="models/gemini-3.5-flash", force_refresh=True)
    )

    assert session.template_calls == 2
    assert first.url.endswith("/1")
    assert cached.url.endswith("/1")
    assert refreshed.url.endswith("/2")


def test_clear_templates_discards_cached_template():
    class _ChangingSession(_FakeSession):
        def __init__(self):
            self.template_calls = 0

        async def capture_template(self, model: str, force_refresh: bool = False):
            self.template_calls += 1
            captured = await super().capture_template(model, force_refresh=force_refresh)
            captured["url"] = f"https://example.test/StreamGenerateContent/{self.template_calls}"
            return captured

    session = _ChangingSession()
    service = RequestCaptureService(session, SnapshotCache())

    first = asyncio.run(service.capture(prompt="hello", model="models/gemini-3.5-flash"))
    service.clear_templates()
    second = asyncio.run(service.capture(prompt="hello", model="models/gemini-3.5-flash"))

    assert session.template_calls == 2
    assert first.url.endswith("/1")
    assert second.url.endswith("/2")
