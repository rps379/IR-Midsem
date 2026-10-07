import argparse
import sys
sys.path.append(".")
from src.corpus import read_corpus
from src.inverted_index import build_inverted_index
from src.weighting import calculate_idf,build_tfidf
from src.retriever import retrieve
from src.pairs import build_claim_evidence_pairs
from src.nli import NLIModel
from src.verdict import verdict_nli
from src.verifier import verify_verdict
from src.explain import explain_trace
from src.reranker import EvidenceReranker


CORPUS_FILE="data/corpus.jsonl"
DOCUMENT_K=10
EVIDENCE_K=5


def build_system(use_reranker=False):
    corpus=read_corpus(
        CORPUS_FILE
    )

    index=build_inverted_index(
        corpus
    )

    idf=calculate_idf(
        index,
        len(corpus)
    )

    tfidf=build_tfidf(
        index,
        idf
    )

    nli_model=NLIModel()

    reranker=None

    if use_reranker:
        reranker=EvidenceReranker()

    return (
        corpus,
        index,
        idf,
        tfidf,
        nli_model,
        reranker
    )


def classify_pairs(
    pairs,
    nli_model
):
    classified=[]

    for pair in pairs:
        result=nli_model.classify(
            pair["claim"],
            pair["evidence"]
        )

        classified_pair=dict(pair)

        classified_pair["label"]=result["label"]
        classified_pair["confidence"]=result["confidence"]
        classified_pair["support_probability"]=result[
            "support_probability"
        ]
        classified_pair["contradict_probability"]=result[
            "contradict_probability"
        ]
        classified_pair["neutral_probability"]=result[
            "neutral_probability"
        ]

        classified.append(
            classified_pair
        )

    return classified


def run_claim(
    claim,
    corpus,
    index,
    idf,
    tfidf,
    nli_model,
    reranker=None
):
    document_results=retrieve(
        claim,
        tfidf,
        idf,
        index,
        DOCUMENT_K
    )

    if not document_results:
        return {
            "verdict":"ABSTAIN",
            "reason":"No relevant documents were retrieved."
        },None

    if reranker is not None:
        pairs=build_claim_evidence_pairs(
            claim,
            document_results,
            corpus,
            idf,
            EVIDENCE_K,
            reranker,
            nli_model
        )

        classified_pairs=pairs

    else:
        pairs=build_claim_evidence_pairs(
            claim,
            document_results,
            corpus,
            idf,
            EVIDENCE_K
        )

        if not pairs:
            return {
                "verdict":"ABSTAIN",
                "reason":"No relevant evidence was retrieved."
            },None

        for pair in pairs:
            pair["claim"]=claim

        classified_pairs=classify_pairs(
            pairs,
            nli_model
        )

    if not classified_pairs:
        return {
            "verdict":"ABSTAIN",
            "reason":"No relevant evidence was retrieved."
        },None

    verdict_result=verdict_nli(
        classified_pairs
    )

    verification_result=verify_verdict(
        classified_pairs,
        verdict_result
    )

    return (
        verification_result,
        {
            "document_results":document_results,
            "pairs":pairs,
            "classified_pairs":classified_pairs,
            "verdict_result":verdict_result,
            "verification_result":verification_result
        }
    )


def main():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--explain",
        action="store_true"
    )

    parser.add_argument(
        "--rerank",
        action="store_true"
    )

    args=parser.parse_args()

    claim=input(
        "Enter scientific claim:\n> "
    ).strip()

    if not claim:
        print("Claim cannot be empty.")
        return

    print()
    print("Loading system...")

    (
        corpus,
        index,
        idf,
        tfidf,
        nli_model,
        reranker
    )=build_system(
        use_reranker=args.rerank
    )

    if args.rerank:
        print(
            "Using BM25 + cross-encoder + NLI reranking."
        )
    else:
        print(
            "Using BM25 evidence retrieval."
        )

    print("Running fact-check...")
    print()

    result,details=run_claim(
        claim,
        corpus,
        index,
        idf,
        tfidf,
        nli_model,
        reranker
    )

    print(
        "FINAL VERDICT:",
        result["verdict"]
    )

    print(
        "Reason:",
        result["reason"]
    )

    if "confidence" in result:
        print(
            f'Confidence: {result["confidence"]:.4f}'
        )

    if "coverage" in result:
        print(
            f'Coverage: {result["coverage"]:.4f}'
        )

    if args.explain and details is not None:
        print()

        print(
            explain_trace(
                claim,
                details["document_results"],
                details["classified_pairs"],
                details["verdict_result"],
                details["verification_result"]
            )
        )


if __name__=="__main__":
    main()