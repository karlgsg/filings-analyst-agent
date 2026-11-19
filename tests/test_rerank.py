from analyst.retrieval.rerank import rerank
from analyst.retrieval.vector import Hit


def test_reranker_puts_relevant_chunk_first():
    hits = [
        Hit(1, "DOC", 1, "Our employees enjoy a collaborative culture and many benefits.", 0.0),
        Hit(2, "DOC", 2, "Purchases of property, plant and equipment were $1,577 million.", 0.0),
        Hit(3, "DOC", 3, "The board of directors met eight times during the year.", 0.0),
    ]
    top = rerank("How much did the company spend on capital expenditures?", hits, k=2)
    assert len(top) == 2
    assert top[0].id == 2
