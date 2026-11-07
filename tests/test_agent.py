"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import (
    agent,
    approve_and_save_learning_log,
    is_learning_log_save_request,
    request_tool_approvals,
)

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_context_is_registered():
    model = TestModel(call_tools=["get_learning_log_context"])
    with agent.override(model=model):
        result = agent.run_sync("学習記録を作るために回答を確認して")

    offered = {tool.name for tool in model.last_model_request_parameters.function_tools}
    called = {part.tool_name for part in _parts(result, ToolCallPart)}
    assert "get_learning_log_context" in offered
    assert "get_learning_log_context" in called
    returns = _parts(result, ToolReturnPart)
    assert any(part.tool_name == "get_learning_log_context" for part in returns)


def test_learning_log_save_request_detection():
    assert is_learning_log_save_request("学習記録を作って保存してください")
    assert is_learning_log_save_request("Save this as a learning log")
    assert not is_learning_log_save_request("学習記録を作ってください")
    assert not is_learning_log_save_request("学習記録は保存しないでください")


def test_cli_fallback_waits_for_approval_before_saving(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr("builtins.input", lambda _: "y")

    result = approve_and_save_learning_log("# 今日の学習\n承認フローを確認した。")

    files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8") == "# 今日の学習\n承認フローを確認した。"
    assert result is not None


def test_cli_fallback_does_not_save_when_denied_or_input_is_closed(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert approve_and_save_learning_log("do not save") is None

    def closed_input(_):
        raise EOFError

    monkeypatch.setattr("builtins.input", closed_input)
    assert approve_and_save_learning_log("still do not save") is None
    assert not tools.LEARNING_LOG_DIR.exists()


def test_main_asks_cli_approval_when_model_returns_text_instead_of_save_tool(
    tmp_path, monkeypatch, capsys
):
    import sys

    from app import agent as agent_module
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    monkeypatch.setattr(
        agent_module,
        "build_model",
        lambda: TestModel(
            call_tools=["get_learning_log_context"],
            custom_output_text="# 学習記録\n承認フローを確認した。",
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.agent", "今回の学習記録を作って保存してください"],
    )
    prompts = []

    def approve(prompt):
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", approve)

    agent_module.main()

    files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(files) == 1
    assert "承認フローを確認した" in files[0].read_text(encoding="utf-8")
    assert prompts == ["この内容を learning_log/ に保存しますか？ [y/N]: "]
    assert "モデルが作成した学習記録案:" in capsys.readouterr().out


def test_learning_log_is_saved_only_after_approval(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("学習記録を保存して")

    assert isinstance(pending.output, DeferredToolRequests)
    assert [call.tool_name for call in pending.output.approvals] == ["save_learning_log"]
    assert not list(tools.LEARNING_LOG_DIR.glob("*.md"))

    monkeypatch.setattr("builtins.input", lambda _: "y")
    deferred_results = request_tool_approvals(pending.output)
    with agent.override(model=TestModel()):
        completed = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=deferred_results,
        )

    assert isinstance(completed.output, str)
    files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(files) == 1


def test_learning_log_is_not_saved_when_approval_is_denied(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    with agent.override(model=TestModel(call_tools=["save_learning_log"])):
        pending = agent.run_sync("学習記録を保存して")

    assert isinstance(pending.output, DeferredToolRequests)
    monkeypatch.setattr("builtins.input", lambda _: "n")
    deferred_results = request_tool_approvals(pending.output)
    with agent.override(model=TestModel()):
        completed = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=deferred_results,
        )

    assert isinstance(completed.output, str)
    assert not list(tools.LEARNING_LOG_DIR.glob("*.md"))
