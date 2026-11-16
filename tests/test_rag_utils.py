from agrisight.rag.chunking import chunk_text
from agrisight.rag.fusion import reciprocal_rank_fusion, tokenize


def test_rrf_prefers_documents_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 1, 4]])
    assert [d for d, _ in fused][:2] in ([1, 2], [2, 1])
    assert fused[-1][0] in (3, 4)


def test_rrf_single_list_keeps_order():
    assert [d for d, _ in reciprocal_rank_fusion([[5, 3, 9]])] == [5, 3, 9]


def test_tokenize():
    assert tokenize("Late-blight, 2nd spray!") == ["late", "blight", "2nd", "spray"]


def test_chunk_text_respects_size_and_keeps_content():
    paras = [f"Paragraph {i} " + "word " * 40 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 50 + 2 for c in chunks)
    for i in range(10):
        assert any(f"Paragraph {i} " in c for c in chunks)


def test_chunk_text_splits_very_long_paragraph():
    chunks = chunk_text("x" * 2000, chunk_size=500, overlap=100)
    assert len(chunks) >= 4
    assert chunk_text("   ") == []
