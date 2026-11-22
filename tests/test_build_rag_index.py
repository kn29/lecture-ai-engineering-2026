"""実SQLiteと固定ベクトルで、索引作成から検索までを確認する。APIは呼ばない。"""

import json
from unittest.mock import Mock

import pytest

from app import retrieval
from scripts.build_rag_index import build_index


@pytest.fixture
def materials(tmp_path):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([
        {"id": f"lec{n:02d}-main", "title": f"講義{n}", "content": [
            {"section": "章", "content": [
                {"id": f"lec{n:02d}-main-p001", "title": f"ページ{n}",
                 "page": 1, "content": f"本文{n}"}]}]}
        for n in (1, 2, 5)
    ], ensure_ascii=False), encoding="utf-8")
    return source, tmp_path / "index.sqlite3"


def test_built_index_can_be_searched(materials, monkeypatch):
    source, database = materials
    texts = []

    def embed(text, task):
        assert task == "RETRIEVAL_DOCUMENT"
        texts.append(text)
        return [1, 0] if "本文1" in text else [0, 1]

    result = build_index(source, database, embedder=embed)
    assert result["pages"] == 2  # 第5回の資料は検索対象に含めない
    assert all("章" in text and "ページ" in text for text in texts)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    hits = retrieval.search("質問")
    assert [h["page_id"] for h in hits if "distance" in h] == [
        "lec01-main-p001", "lec02-main-p001",
    ]
    assert hits[0]["content"] == "本文1"
    assert hits[0]["distance"] == pytest.approx(0)


def test_search_neighbors_stay_in_document_and_are_deduplicated(tmp_path):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([
        {"id": "lec01-main", "title": "講義1", "content": [
            {"section": "章", "content": [
                {"id": f"lec01-main-p00{page}", "title": f"ページ{page}",
                 "page": page, "content": f"本文{page}"}
                for page in (1, 2, 3)]}]},
        {"id": "lec02-main", "title": "講義2", "content": [
            {"section": "章", "content": [
                {"id": "lec02-main-p001", "title": "別資料のページ1",
                 "page": 1, "content": "別資料の本文"}]}]},
    ], ensure_ascii=False), encoding="utf-8")
    results = [
        {"page_id": "lec01-main-p002", "document_id": "lec01-main",
         "page": 2, "content": "本文2", "distance": 0.1},
        {"page_id": "lec01-main-p003", "document_id": "lec01-main",
         "page": 3, "content": "本文3", "distance": 0.2},
    ]

    expanded = retrieval.add_neighbor_pages(results, source)

    assert [page["page_id"] for page in expanded] == [
        "lec01-main-p002", "lec01-main-p003", "lec01-main-p001",
    ]
    assert expanded[1]["distance"] == 0.2
    assert expanded[2]["relative_position"] == "previous"
    assert expanded[2]["neighbor_of"] == "lec01-main-p002"
    assert "distance" not in expanded[2]


def test_graph_uses_vector_matches_as_seeds(tmp_path, monkeypatch):
    root = {"page_id": "lec01-main-p001", "document_id": "lec01-main",
            "lecture": 1, "document_title": "講義1", "section": "章",
            "title": "検索結果", "page": 1, "content": "本文"}
    neighbor = {**root, "page_id": "lec01-main-p002", "page": 2}
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({
        "nodes": [
            {"id": root["page_id"], "kind": "page", "title": root["title"]},
            {"id": neighbor["page_id"], "kind": "page", "title": neighbor["title"]},
        ],
        "edges": [],
    }), encoding="utf-8")
    monkeypatch.setattr(retrieval, "load_pages", lambda *_: [root, neighbor])
    monkeypatch.setattr(retrieval, "RAG_GRAPH", graph_path)
    monkeypatch.setattr(retrieval, "search", lambda _: [
        {**root, "distance": 0.1},
        {**neighbor, "retrieval": "neighbor"},
    ])

    result = retrieval.search_graph("質問")

    assert [page["page_id"] for page in result["seeds"]] == [root["page_id"]]


def test_existing_index_survives_failed_rebuild(materials):
    source, database = materials
    build_index(source, database, embedder=lambda *_: [1, 0])
    before = database.read_bytes()
    embed = Mock(side_effect=RuntimeError("埋め込みに失敗"))
    with pytest.raises(FileExistsError):
        build_index(source, database, embedder=embed)
    embed.assert_not_called()
    with pytest.raises(RuntimeError, match="埋め込みに失敗"):
        build_index(source, database, rebuild=True, embedder=embed)
    assert database.read_bytes() == before
    assert not list(database.parent.glob(".rag-build-*"))
