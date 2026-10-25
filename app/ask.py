"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"

_CLIENT = None
_CHAT_SESSION = None


def get_client():
    """再利用する GenAI クライアントを返す。"""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=get_api_key())
    return _CLIENT


def get_chat_session():
    """会話履歴を保持するチャットセッションを返す。"""
    global _CHAT_SESSION
    if _CHAT_SESSION is None:
        client = get_client()
        _CHAT_SESSION = client.chats.create(model=MODEL)
    return _CHAT_SESSION


def ask(question: str) -> str:
    chat = get_chat_session()
    response = chat.send_message(question)

    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None)
        output_tokens = getattr(usage, "candidates_token_count", None)
        print(f"[入力トークン数] {input_tokens}")
        print(f"[出力トークン数] {output_tokens}")

    return response.text


def measure_response(model_name: str, question: str, temperature: float | None = None) -> dict:
    """応答時間とトークン数をまとめて測る。"""
    client = genai.Client(api_key=get_api_key())
    config = None if temperature is None else types.GenerateContentConfig(temperature=temperature)

    start = time.perf_counter()
    response = client.models.generate_content(model=model_name, contents=question, config=config)
    elapsed = time.perf_counter() - start

    usage = getattr(response, "usage_metadata", None)
    input_tokens = getattr(usage, "prompt_token_count", None)
    output_tokens = getattr(usage, "candidates_token_count", None)
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "model": model_name,
        "question": question,
        "temperature": temperature,
        "elapsed_sec": round(elapsed, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "answer": response.text,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保ちながらLLMに質問する")
    parser.add_argument("question", nargs="?", default=None)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.question is not None:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
            return

        while True:
            question = input("質問を入力してください（終了は q または quit）> ")
            if question.strip().lower() in {"q", "quit", "exit"}:
                print("会話を終了します。")
                return
            print(f"[質問] {question}\n")
            print(ask(question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
