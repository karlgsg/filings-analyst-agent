import pytest

from analyst.ingest.chunker import chunk_pages, split_words


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert split_words(words(10), size=50, overlap=5) == [words(10)]


def test_empty_text_gives_no_chunks():
    assert split_words("   \n ", size=50, overlap=5) == []


def test_overlap_repeats_words_between_chunks():
    pieces = split_words(words(25), size=10, overlap=3)
    assert pieces[0].split()[-3:] == pieces[1].split()[:3]


def test_every_word_is_covered():
    text = words(1000)
    pieces = split_words(text, size=100, overlap=20)
    covered = {w for p in pieces for w in p.split()}
    assert covered == set(text.split())


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        split_words(words(10), size=10, overlap=10)


def test_chunks_keep_page_numbers_and_never_cross_pages():
    pages = ["alpha " * 5, "", "beta " * 5]
    chunks = chunk_pages("DOC", pages)
    assert [(c.page, c.text.split()[0]) for c in chunks] == [(1, "alpha"), (3, "beta")]
    assert all(c.doc_name == "DOC" for c in chunks)
