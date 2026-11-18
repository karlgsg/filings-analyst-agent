import numpy as np

from analyst.embeddings import DIM, embed_documents, embed_query


def test_embeddings_have_right_shape_and_unit_length():
    docs = embed_documents(["Capital expenditures were $1,577 million.", "The cat sat."])
    assert docs.shape == (2, DIM)
    assert np.allclose(np.linalg.norm(docs, axis=1), 1.0, atol=1e-3)


def test_query_is_closer_to_related_text():
    docs = embed_documents(
        [
            "Purchases of property, plant and equipment totaled $1,577 million.",
            "Our employees enjoy a collaborative culture.",
        ]
    )
    q = embed_query("What was capital expenditure?")
    sims = docs @ q  # dot product == cosine similarity for unit vectors
    assert sims[0] > sims[1]
