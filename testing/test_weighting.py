import math
import sys
sys.path.append(".")
from src.weighting import calculate_idf,build_tfidf


def test_idf():
    index={
        "vitamin":{
            "df":2,
            "postings":[
                {"docid":1,"tf":2},
                {"docid":2,"tf":1}
            ]
        },
        "bone":{
            "df":1,
            "postings":[
                {"docid":1,"tf":2}
            ]
        }
    }

    idf=calculate_idf(index,2)

    assert idf["vitamin"]==math.log(3/3)+1
    assert idf["bone"]==math.log(3/2)+1


def test_tfidf():
    index={
        "bone":{
            "df":1,
            "postings":[
                {"docid":1,"tf":2}
            ]
        }
    }

    idf=calculate_idf(index,2)
    tfidf=build_tfidf(index,idf)

    expected=(1+math.log(2))*idf["bone"]

    assert math.isclose(tfidf[1]["bone"],expected)