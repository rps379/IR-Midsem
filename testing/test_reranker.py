import time
import sys
sys.path.append(".")
from src.corpus import read_corpus,read_claims
from src.inverted_index import build_inverted_index
from src.weighting import calculate_idf,build_tfidf
from src.retriever import retrieve
from src.evidence import retrieve_evidence
from src.reranker import EvidenceReranker


CORPUS_FILE="data/corpus.jsonl"
CLAIMS_FILE="data/claims_dev.jsonl"

DOCUMENT_K=10
CANDIDATE_K=15


def get_gold_evidence(claim):
    gold=set()

    for docid,evidence_list in claim["evidence"].items():
        for evidence in evidence_list:
            for sentence_id in evidence["sentences"]:
                gold.add(
                    (
                        int(docid),
                        sentence_id
                    )
                )

    return gold


def get_candidates(
    claim_text,
    document_results,
    corpus,
    idf
):
    candidates=[]

    for docid,document_score in document_results:
        document=corpus[docid]

        evidence_results=retrieve_evidence(
            claim_text,
            document,
            idf,
            CANDIDATE_K
        )

        for evidence in evidence_results:
            candidates.append({
                "claim":claim_text,
                "docid":docid,
                "sentence_id":evidence["sentence_id"],
                "evidence":evidence["text"],
                "document_score":document_score,
                "evidence_score":evidence["score"],
                "bm25_score":evidence.get(
                    "bm25_score",
                    evidence["score"]
                )
            })

    return candidates


def rank_bm25(candidates,k):
    results=sorted(
        candidates,
        key=lambda candidate:(
            -candidate["bm25_score"],
            candidate["docid"],
            candidate["sentence_id"]
        )
    )

    return results[:k]


def calculate_recall(results,gold,k):
    if not gold:
        return 0.0

    retrieved=set()

    for result in results[:k]:
        retrieved.add(
            (
                result["docid"],
                result["sentence_id"]
            )
        )

    return len(
        retrieved & gold
    )/len(gold)


def main():
    start_time=time.time()

    print(
        "Cross-Encoder Evidence Retrieval"
    )

    print(
        "---------------------------------"
    )

    print(
        "Loading corpus..."
    )

    corpus=read_corpus(
        CORPUS_FILE
    )

    print(
        f"Loaded {len(corpus)} documents."
    )

    print(
        "Loading claims..."
    )

    claims=read_claims(
        CLAIMS_FILE
    )

    claims=[
        claim
        for claim in claims
        if "evidence" in claim
    ]

    print(
        f"Loaded {len(claims)} evaluated claims."
    )

    print(
        "Building inverted index..."
    )

    index=build_inverted_index(
        corpus
    )

    print(
        "Calculating IDF..."
    )

    idf=calculate_idf(
        index,
        len(corpus)
    )

    print(
        "Building TF-IDF..."
    )

    tfidf=build_tfidf(
        index,
        idf
    )

    print(
        "Loading cross-encoder..."
    )

    reranker=EvidenceReranker()

    print(
        "Cross-encoder loaded."
    )

    print()

    bm25_totals={
        1:0.0,
        3:0.0,
        5:0.0,
        10:0.0
    }

    reranker_totals={
        1:0.0,
        3:0.0,
        5:0.0,
        10:0.0
    }

    evaluated=0

    total_claims=len(claims)

    for claim_number,claim in enumerate(
        claims,
        start=1
    ):
        claim_start=time.time()

        document_results=retrieve(
            claim["claim"],
            tfidf,
            idf,
            index,
            DOCUMENT_K
        )

        if not document_results:
            print(
                f"[{claim_number}/{total_claims}] "
                f"No documents retrieved."
            )
            continue

        candidates=get_candidates(
            claim["claim"],
            document_results,
            corpus,
            idf
        )

        if not candidates:
            print(
                f"[{claim_number}/{total_claims}] "
                f"No evidence candidates."
            )
            continue

        gold=get_gold_evidence(
            claim
        )

        bm25_results=rank_bm25(
            candidates,
            10
        )

        reranked_results=reranker.rerank(
            claim["claim"],
            candidates,
            10
        )

        for k in [1,3,5,10]:
            bm25_totals[k]+=calculate_recall(
                bm25_results,
                gold,
                k
            )

            reranker_totals[k]+=calculate_recall(
                reranked_results,
                gold,
                k
            )

        evaluated+=1

        elapsed=time.time()-claim_start

        print(
            f"[{claim_number}/{total_claims}] "
            f"evaluated in {elapsed:.2f}s"
        )

    total_time=time.time()-start_time

    print()

    print(
        "Cross-Encoder Evidence Retrieval"
    )

    print(
        "---------------------------------"
    )

    print(
        f"Evaluated claims: {evaluated}"
    )

    print(
        f"Total time: {total_time:.2f}s"
    )

    if evaluated==0:
        print(
            "No claims were evaluated."
        )
        return

    print()

    print(
        "Evidence Recall"
    )

    print(
        "---------------"
    )

    for k in [1,3,5,10]:
        bm25_recall=(
            bm25_totals[k]/evaluated
        )

        reranker_recall=(
            reranker_totals[k]/evaluated
        )

        print(
            f"Recall@{k}: "
            f"BM25={bm25_recall:.4f} "
            f"CrossEncoder={reranker_recall:.4f}"
        )

    print()

    print(
        "Improvement"
    )

    print(
        "-----------"
    )

    for k in [1,3,5,10]:
        bm25_recall=(
            bm25_totals[k]/evaluated
        )

        reranker_recall=(
            reranker_totals[k]/evaluated
        )

        improvement=(
            reranker_recall-
            bm25_recall
        )

        print(
            f"Recall@{k}: "
            f"{improvement:+.4f}"
        )


if __name__=="__main__":
    main()