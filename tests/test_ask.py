from types import SimpleNamespace

from app import ask as ask_module


class StubResponse:
    def __init__(self, text: str):
        self.text = text
        self.usage_metadata = SimpleNamespace(
            prompt_token_count=7,
            candidates_token_count=12,
        )


class StubChat:
    def __init__(self):
        self.messages = []

    def send_message(self, question: str):
        self.messages.append(question)
        return StubResponse(f"answer:{question}")


class StubClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.chats = SimpleNamespace(create=self.create_chat)
        self.chat = StubChat()

    def create_chat(self, model: str):
        return self.chat


def test_ask_reuses_the_same_chat_session(monkeypatch):
    calls = []

    def fake_client(api_key: str):
        calls.append(api_key)
        return StubClient(api_key)

    monkeypatch.setattr(ask_module.genai, "Client", fake_client)
    monkeypatch.setattr(ask_module, "_CLIENT", None)
    monkeypatch.setattr(ask_module, "_CHAT_SESSION", None)

    first = ask_module.ask("おはよう")
    second = ask_module.ask("続けて質問")

    assert first == "answer:おはよう"
    assert second == "answer:続けて質問"
    assert len(calls) == 1
    assert ask_module._CLIENT is not None
    assert ask_module._CHAT_SESSION is not None


def test_measure_response_returns_elapsed_time_and_token_counts(monkeypatch):
    class DummyResponse:
        def __init__(self):
            self.text = "response-text"
            self.usage_metadata = SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=20,
            )

    class DummyModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config=None):
            self.calls.append((model, contents, getattr(config, "temperature", None)))
            return DummyResponse()

    class DummyClient:
        def __init__(self):
            self.models = DummyModels()

    times = iter([100.0, 100.25])
    monkeypatch.setattr(ask_module.time, "perf_counter", lambda: next(times))
    monkeypatch.setattr(ask_module.genai, "Client", lambda api_key: DummyClient())

    stats = ask_module.measure_response("gemini-3.5-flash", "テスト質問", temperature=0.7)

    assert stats["model"] == "gemini-3.5-flash"
    assert stats["question"] == "テスト質問"
    assert stats["temperature"] == 0.7
    assert stats["elapsed_sec"] == 0.25
    assert stats["input_tokens"] == 10
    assert stats["output_tokens"] == 20
    assert stats["total_tokens"] == 30
