import asyncio

from aistudio_api.api.schemas import ChatRequest
from aistudio_api.application.api_service import handle_chat


class _CaptureChatClient:
    def __init__(self):
        self.calls = []

    async def generate_content(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})

        class _Output:
            text = "ok"
            thinking = ""
            usage = {}
            function_calls = []

        return _Output()


def test_handle_chat_empty_tools_disables_model_defaults(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemma-4-31b-it",
        messages=[{"role": "user", "content": "hello"}],
        tools=[],
    )

    asyncio.run(handle_chat(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["tools"] == []


def test_handle_chat_reenables_tools_after_new_user_message(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[
            {"role": "user", "content": "run date"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"date +%s"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"output":"123"}'},
            {"role": "user", "content": "进度如何？继续执行"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )

    asyncio.run(handle_chat(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["tools"] != []


def test_handle_chat_keeps_explicit_tools_after_tool_result(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[
            {"role": "user", "content": "run date"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"command":"date +%s"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"output":"123"}'},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                    },
                },
            }
        ],
    )

    asyncio.run(handle_chat(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["tools"] != []


def test_normalize_chat_request_tool_calls_and_responses():
    from aistudio_api.application.chat_service import normalize_chat_request
    from aistudio_api.api.schemas import Message

    messages = [
        Message(role="user", content="What is the weather like in Beijing?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Beijing"}'},
                }
            ],
        ),
        Message(
            role="tool",
            tool_call_id="call_123",
            name="get_weather",
            content='{"temperature": 24, "condition": "sunny"}',
        ),
    ]

    res = normalize_chat_request(messages, "gemini-3.5-flash")
    contents = res["contents"]

    assert len(contents) == 3

    # Check User Message
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "What is the weather like in Beijing?"

    # Completed tool calls/results use a text compatibility path instead of
    # replaying native function_response parts, while preserving both sides of
    # the history for agent clients.
    assert contents[1].role == "model"
    assert '<tool_call name="get_weather" tool_call_id="call_123">' in contents[1].parts[0].text
    assert '{"location": "Beijing"}' in contents[1].parts[0].text

    assert contents[2].role == "user"
    assert '<tool_result name="get_weather" tool_call_id="call_123">' in contents[2].parts[0].text
    assert '{"temperature": 24, "condition": "sunny"}' in contents[2].parts[0].text


def test_normalize_chat_request_preserves_assistant_tool_call_before_tool_result():
    from aistudio_api.application.chat_service import normalize_chat_request
    from aistudio_api.api.schemas import Message

    messages = [
        Message(role="user", content="What is the weather like in Beijing?"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Beijing"}'},
                }
            ],
        ),
    ]

    contents = normalize_chat_request(messages, "gemini-3.5-flash")["contents"]

    assert len(contents) == 2
    assert contents[1].role == "model"
    assert contents[1].parts[0].function_call == ("get_weather", {"location": "Beijing"}, "call_123")



def test_handle_chat_applies_openai_env_defaults(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_temperature", 0.6)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_top_p", 0.95)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_top_k", 40)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_max_tokens", 32768)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_thinking", "high")
    monkeypatch.setattr(api_service_openai.settings, "openai_default_safety_off", True)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_google_search", False)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
    )

    asyncio.run(handle_chat(req, client))

    kwargs = client.calls[0]["kwargs"]
    assert kwargs["temperature"] == 0.6
    assert kwargs["top_p"] == 0.95
    assert kwargs["top_k"] == 40
    assert kwargs["max_tokens"] == 32768
    assert kwargs["safety_settings"] == [[None, None, 7, 5], [None, None, 8, 5], [None, None, 9, 5], [None, None, 10, 5]]
    assert kwargs["generation_config_overrides"] == {"thinking_config": [1, None, None, 3]}


def test_handle_chat_request_values_override_openai_env_defaults(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_temperature", 0.6)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_top_p", 0.95)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_top_k", 40)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_max_tokens", 32768)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_thinking", "high")
    monkeypatch.setattr(api_service_openai.settings, "openai_default_safety_off", True)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_google_search", True)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.2,
        top_p=0.7,
        top_k=12,
        max_tokens=1024,
        thinking="off",
        safety_off=False,
        google_search=False,
    )

    asyncio.run(handle_chat(req, client))

    kwargs = client.calls[0]["kwargs"]
    assert kwargs["temperature"] == 0.2
    assert kwargs["top_p"] == 0.7
    assert kwargs["top_k"] == 12
    assert kwargs["max_tokens"] == 1024
    assert kwargs["safety_settings"] is None
    assert kwargs["generation_config_overrides"] == {"thinking_config": None}
    assert kwargs["tools"] is None


def test_handle_chat_env_google_search_not_auto_added_to_explicit_agent_tools(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_google_search", True)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[{"role": "user", "content": "hello"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "terminal",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                },
            }
        ],
    )

    asyncio.run(handle_chat(req, client))

    tools = client.calls[0]["kwargs"]["tools"]
    assert tools is not None
    assert len(tools) == 1


def test_handle_chat_env_google_search_skipped_for_context_summarization(monkeypatch):
    from aistudio_api.application import api_service_openai

    async def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr(api_service_openai, "require_busy_lock", lambda: asyncio.Semaphore(1))
    monkeypatch.setattr(api_service_openai, "ensure_active_account", _noop)
    monkeypatch.setattr(api_service_openai, "record_rotator_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(api_service_openai.settings, "openai_default_google_search", True)

    client = _CaptureChatClient()
    req = ChatRequest(
        model="gemini-3.5-flash",
        messages=[
            {
                "role": "user",
                "content": "You are a summarization agent creating a context summary for later turns.",
            }
        ],
    )

    asyncio.run(handle_chat(req, client))

    assert len(client.calls) == 1
    assert client.calls[0]["kwargs"]["tools"] is None


def test_extract_pseudo_tool_call_from_attribute_arguments():
    from aistudio_api.application.api_service_openai import _extract_pseudo_tool_calls

    text = '''<tool_call name="execute_code" code="
import json

path = \"/var/home/deck/solutions.json\"
with open(path, 'r', encoding='utf-8') as f:
    print(f.read(10))
}"}
]'''

    calls = _extract_pseudo_tool_calls(text)

    assert calls == [
        {
            "name": "execute_code",
            "args": {
                "code": "\nimport json\n\npath = \"/var/home/deck/solutions.json\"\nwith open(path, 'r', encoding='utf-8') as f:\n    print(f.read(10))"
            },
        }
    ]


def test_detect_incomplete_final_text_markdown_table():
    from aistudio_api.application.api_service_openai import _detect_incomplete_final_text

    incomplete = """| 标题 | 链接 | 摘要 |
| --- | --- | --- |
| item1 | https://example.test/topic/1 |"""

    complete = """| 标题 | 链接 | 摘要 |
| --- | --- | --- |
| item1 | https://example.test/topic/1 | 完整摘要 |"""

    assert _detect_incomplete_final_text(incomplete) == "incomplete_markdown_table_row"
    assert _detect_incomplete_final_text(complete) is None


def test_detect_incomplete_final_text_pseudo_tool_call_tag():
    from aistudio_api.application.api_service_openai import _detect_incomplete_final_text

    tag = "<" + "tool_call"
    partial = tag + ' name="browser_scroll" tool_call_id="abc123'
    unclosed = tag + ' name="browser_scroll">{"direction":"down"}'
    complete = tag + ' name="browser_scroll">{"direction":"down"}</' + "tool_call" + ">"

    assert _detect_incomplete_final_text(partial) == "incomplete_pseudo_tool_call_tag"
    assert _detect_incomplete_final_text(unclosed) == "unclosed_pseudo_tool_call"
    assert _detect_incomplete_final_text(complete) is None


def test_maybe_continue_incomplete_final_text_requests_one_text_only_continuation():
    from aistudio_api.application.api_service_openai import _maybe_continue_incomplete_final_text
    from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart

    class _ContinuationClient:
        def __init__(self):
            self.calls = []

        async def stream_generate_content(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            yield "body", " 继续的表格内容 |\n"
            yield "usage", {"completion_tokens": 3}

    client = _ContinuationClient()
    partial = """| 标题 | 链接 | 摘要 |
| --- | --- | --- |
| item1 | https://example.test/topic/1 |"""

    continuation = asyncio.run(
        _maybe_continue_incomplete_final_text(
            client=client,
            model="gemini-3.5-flash",
            capture_prompt="prompt",
            capture_images=None,
            contents=[AistudioContent(role="user", parts=[AistudioPart(text="列出item1结果")])],
            system_instruction=None,
            partial_text=partial,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            safety_settings=None,
            generation_config_overrides=None,
        )
    )

    assert continuation == " 继续的表格内容 |\n"
    assert len(client.calls) == 1
    kwargs = client.calls[0]["kwargs"]
    assert kwargs["tools"] is None
    assert len(kwargs["contents"]) == 1
    assert kwargs["contents"][0].role == "user"
    repair_prompt = kwargs["contents"][0].parts[0].text
    assert "列出item1结果" in repair_prompt
    assert partial in repair_prompt
    assert "不要调用工具" in repair_prompt


def test_maybe_continue_incomplete_final_text_ignores_complete_table():
    from aistudio_api.application.api_service_openai import _maybe_continue_incomplete_final_text
    from aistudio_api.infrastructure.gateway.wire_types import AistudioContent, AistudioPart

    class _ContinuationClient:
        def __init__(self):
            self.calls = []

        async def stream_generate_content(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            yield "body", "should not be called"

    client = _ContinuationClient()
    complete = """| 标题 | 链接 | 摘要 |
| --- | --- | --- |
| item1 | https://example.test/topic/1 | 完整摘要 |"""

    continuation = asyncio.run(
        _maybe_continue_incomplete_final_text(
            client=client,
            model="gemini-3.5-flash",
            capture_prompt="prompt",
            capture_images=None,
            contents=[AistudioContent(role="user", parts=[AistudioPart(text="列出item1结果")])],
            system_instruction=None,
            partial_text=complete,
            temperature=None,
            top_p=None,
            top_k=None,
            max_tokens=None,
            safety_settings=None,
            generation_config_overrides=None,
        )
    )

    assert continuation == ""
    assert client.calls == []
