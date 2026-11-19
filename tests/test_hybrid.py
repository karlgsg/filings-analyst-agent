from analyst.retrieval.hybrid import rrf_fuse
from analyst.retrieval.vector import Hit


def hit(i: int) -> Hit:
    return Hit(id=i, doc_name="DOC", page=i, text=f"chunk {i}", score=0.0)


def test_chunk_in_both_lists_beats_chunk_in_one():
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(4), hit(2), hit(5)]
    fused = rrf_fuse([vector, keyword])
    assert fused[0].id == 2  # rank 2 in both lists beats rank 1 in only one


def test_rrf_scores_are_sum_of_reciprocal_ranks():
    fused = rrf_fuse([[hit(1)], [hit(1)]], k=60)
    assert abs(fused[0].score - 2 / 61) < 1e-12


def test_every_chunk_kept_once():
    fused = rrf_fuse([[hit(1), hit(2)], [hit(2), hit(3)]])
    assert sorted(h.id for h in fused) == [1, 2, 3]


def test_empty_lists():
    assert rrf_fuse([[], []]) == []
