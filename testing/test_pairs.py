from src.corpus import read_corpus,read_claims
from src.inverted_index import build_inverted_index
from src.weighting import calculate_idf,build_tfidf
from src.retriever import retrieve
from src.pairs import build_claim_evidence_pairs


corpus=read_corpus("data/corpus.jsonl")
claims=read_claims("data/claims_dev.jsonl")

index=build_inverted_index(corpus)

idf=calculate_idf(index,len(corpus))

tfidf=build_tfidf(index,idf)


def test_claim_evidence_pairs():

    claim=next(
        claim
        for claim in claims
        if "evidence" in claim
    )

    document_results=retrieve(
        claim["claim"],
        tfidf,
        idf,
        index,
        10
    )

    pairs=build_claim_evidence_pairs(
        claim["claim"],
        document_results,
        corpus,
        idf,
        5
    )

    assert isinstance(pairs,list)
    assert len(pairs)<=50

    for pair in pairs:
        assert pair["claim"]==claim["claim"]

        assert isinstance(
            pair["docid"],
            int
        )

        assert isinstance(
            pair["sentence_id"],
            int
        )

        assert isinstance(
            pair["evidence"],
            str
        )

        assert isinstance(
            pair["document_score"],
            float
        )

        assert isinstance(
            pair["evidence_score"],
            float
        )

        assert pair["docid"] in corpus

        assert 0<=pair["sentence_id"]<len(
            corpus[pair["docid"]]["abstract"]
        )

        assert pair["evidence"]==(
            corpus[pair["docid"]]["abstract"][
                pair["sentence_id"]
            ]
        )


def test_pair_count_per_document():

    claim=next(
        claim
        for claim in claims
        if "evidence" in claim
    )

    document_results=retrieve(
        claim["claim"],
        tfidf,
        idf,
        index,
        3
    )

    pairs=build_claim_evidence_pairs(
        claim["claim"],
        document_results,
        corpus,
        idf,
        5
    )

    for docid,document_score in document_results:

        document_pairs=[
            pair
            for pair in pairs
            if pair["docid"]==docid
        ]

        assert len(document_pairs)<=5


def test_pairs_preserve_retrieval_order():

    claim=next(
        claim
        for claim in claims
        if "evidence" in claim
    )

    document_results=retrieve(
        claim["claim"],
        tfidf,
        idf,
        index,
        3
    )

    pairs=build_claim_evidence_pairs(
        claim["claim"],
        document_results,
        corpus,
        idf,
        5
    )

    retrieved_docids=[
        docid
        for docid,score in document_results
    ]

    pair_docids=[]

    for pair in pairs:
        if pair["docid"] not in pair_docids:
            pair_docids.append(pair["docid"])

    assert pair_docids==retrieved_docids