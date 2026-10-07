import sys
sys.path.append(".")
from src.inverted_index import build_inverted_index


def test_term_exists():
    corpus={
        1:{"abstract":["Vitamin D improves bone health"]},
        2:{"abstract":["Bone health is important"]}
    }

    index=build_inverted_index(corpus)

    assert "vitamin" in index
    assert "bone" in index


def test_document_frequency():
    corpus={
        1:{"abstract":["Vitamin D improves bone health"]},
        2:{"abstract":["Bone health is important"]}
    }

    index=build_inverted_index(corpus)

    assert index["bone"]["df"]==2
    assert index["vitamin"]["df"]==1


def test_term_frequency():
    corpus={
        1:{"abstract":["Vitamin D and vitamin D"]},
    }

    index=build_inverted_index(corpus)

    assert index["vitamin"]["postings"][0]["tf"]==2


def test_postings():
    corpus={
        1:{"abstract":["Vitamin D"]},
        2:{"abstract":["Vitamin C"]}
    }

    index=build_inverted_index(corpus)

    docids=[posting["docid"] for posting in index["vitamin"]["postings"]]

    assert docids==[1,2]