import sys
sys.path.append(".")
from src.preprocess import preprocess


def test_lowercase():
    assert preprocess("Vitamin") == ["vitamin"]


def test_stopwords():
    result=preprocess("the study of vitamin")
    assert "the" not in result
    assert "of" not in result


def test_vitamin_d():
    result=preprocess("Vitamin D improves bone health")
    assert "vitamin" in result
    assert "d" in result


def test_punctuation():
    result=preprocess("health, bone.")
    assert "health" in result
    assert "bone" in result


def test_lemmatization():
    result=preprocess("studies cells")
    assert "study" in result
    assert "cell" in result