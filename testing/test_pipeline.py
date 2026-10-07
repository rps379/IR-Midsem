from src.corpus import read_corpus,read_claims
from src.inverted_index import build_inverted_index
from src.weighting import calculate_idf,build_tfidf
from src.retriever import retrieve
from src.evidence import retrieve_evidence


corpus=read_corpus("data/corpus.jsonl")
claims=read_claims("data/claims_dev.jsonl")

index=build_inverted_index(corpus)

idf=calculate_idf(index,len(corpus))

tfidf=build_tfidf(index,idf)


def document_recall_at_k(results,gold_docids,k):
    retrieved={docid for docid,score in results[:k]}
    gold=set(gold_docids)

    if not gold:
        return 0

    return len(retrieved&gold)/len(gold)


def evidence_recall_at_k(
    claim,
    document_results,
    document_k,
    evidence_k
):
    retrieved_documents={
        docid
        for docid,score in document_results[:document_k]
    }

    gold_evidence=set()

    for docid,evidence_list in claim["evidence"].items():
        docid=int(docid)

        for evidence in evidence_list:
            for sentence_id in evidence["sentences"]:
                gold_evidence.add(
                    (docid,sentence_id)
                )

    if not gold_evidence:
        return 0

    retrieved_evidence=set()

    for docid in retrieved_documents:
        results=retrieve_evidence(
            claim["claim"],
            corpus[docid],
            idf,
            evidence_k
        )

        for result in results:
            retrieved_evidence.add(
                (docid,result["sentence_id"])
            )

    return len(retrieved_evidence&gold_evidence)/len(gold_evidence)


def test_end_to_end_evidence_pipeline():
    document_ks=[1,3,5,10]
    evidence_ks=[1,3,5]

    document_totals={k:0 for k in document_ks}

    evidence_totals={
        (document_k,evidence_k):0
        for document_k in document_ks
        for evidence_k in evidence_ks
    }

    count=0

    for claim in claims:
        if "evidence" not in claim:
            continue

        gold_docids=[
            int(docid)
            for docid in claim["evidence"]
        ]

        if not gold_docids:
            continue

        document_results=retrieve(
            claim["claim"],
            tfidf,
            idf,
            index,
            max(document_ks)
        )

        for document_k in document_ks:
            document_totals[document_k]+=document_recall_at_k(
                document_results,
                gold_docids,
                document_k
            )

            for evidence_k in evidence_ks:
                evidence_totals[
                    (document_k,evidence_k)
                ]+=evidence_recall_at_k(
                    claim,
                    document_results,
                    document_k,
                    evidence_k
                )

        count+=1

    assert count>0

    print()
    print("End-to-End Evidence Retrieval")
    print("------------------------------")
    print("Evaluated claims:",count)
    print()

    print("Document Retrieval")
    print("------------------")

    for k in document_ks:
        recall=document_totals[k]/count
        print(f"Recall@{k}: {recall:.4f}")

    print()

    print("End-to-End Evidence Recall")
    print("---------------------------")

    for document_k in document_ks:
        print(f"Top-{document_k} Documents")

        for evidence_k in evidence_ks:
            recall=evidence_totals[
                (document_k,evidence_k)
            ]/count

            print(
                f"Evidence Recall@{evidence_k}: {recall:.4f}"
            )

        print()


def test_conditional_evidence_recall():
    document_k=10
    evidence_ks=[1,3,5]

    totals={k:0 for k in evidence_ks}

    retrieved_document_count=0
    total_gold_pairs=0

    for claim in claims:
        if "evidence" not in claim:
            continue

        gold_evidence=set()

        for docid,evidence_list in claim["evidence"].items():
            docid=int(docid)

            for evidence in evidence_list:
                for sentence_id in evidence["sentences"]:
                    gold_evidence.add(
                        (docid,sentence_id)
                    )

        if not gold_evidence:
            continue

        document_results=retrieve(
            claim["claim"],
            tfidf,
            idf,
            index,
            document_k
        )

        retrieved_documents={
            docid
            for docid,score in document_results
        }

        gold_documents={
            docid
            for docid,sentence_id in gold_evidence
        }

        relevant_documents=(
            retrieved_documents&gold_documents
        )

        if not relevant_documents:
            continue

        retrieved_document_count+=1

        for docid in relevant_documents:
            total_gold_pairs+=sum(
                1
                for gold_docid,gold_sentence_id
                in gold_evidence
                if gold_docid==docid
            )

        for evidence_k in evidence_ks:
            retrieved_evidence=set()

            for docid in relevant_documents:
                results=retrieve_evidence(
                    claim["claim"],
                    corpus[docid],
                    idf,
                    evidence_k
                )

                for result in results:
                    retrieved_evidence.add(
                        (docid,result["sentence_id"])
                    )

            relevant_gold_evidence=(
                gold_evidence
                & {
                    (docid,sentence_id)
                    for docid in relevant_documents
                    for sentence_id in range(
                        len(corpus[docid]["abstract"])
                    )
                }
            )

            if relevant_gold_evidence:
                totals[evidence_k]+=(
                    len(
                        retrieved_evidence
                        & relevant_gold_evidence
                    )
                    /
                    len(relevant_gold_evidence)
                )

    assert retrieved_document_count>0

    print()
    print("Conditional Evidence Retrieval")
    print("-------------------------------")
    print(
        "Claims with gold document in Top-10:",
        retrieved_document_count
    )
    print()

    for k in evidence_ks:
        recall=totals[k]/retrieved_document_count

        print(
            f"Evidence Recall@{k} "
            f"given correct document: {recall:.4f}"
        )


def test_evidence_results_are_valid():
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

    assert len(document_results)<=3

    for docid,score in document_results:
        results=retrieve_evidence(
            claim["claim"],
            corpus[docid],
            idf,
            5
        )

        assert len(results)<=5

        for result in results:
            sentence_id=result["sentence_id"]

            assert isinstance(sentence_id,int)
            assert isinstance(result["text"],str)
            assert isinstance(result["score"],float)
            assert 0<=sentence_id<len(corpus[docid]["abstract"])
            assert result["text"]==corpus[docid]["abstract"][sentence_id]