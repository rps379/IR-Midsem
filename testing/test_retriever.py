from src.corpus import read_corpus,read_claims
from src.inverted_index import build_inverted_index
from src.weighting import calculate_idf,build_tfidf
from src.retriever import retrieve


corpus=read_corpus("data/corpus.jsonl")
claims=read_claims("data/claims_dev.jsonl")

index=build_inverted_index(corpus)

idf=calculate_idf(index,len(corpus))

tfidf=build_tfidf(index,idf)


def recall_at_k(results,gold_docids,k):
    retrieved={docid for docid,score in results[:k]}
    gold=set(gold_docids)

    if not gold:
        return 0

    return len(retrieved&gold)/len(gold)


def precision_at_k(results,gold_docids,k):
    retrieved={docid for docid,score in results[:k]}
    gold=set(gold_docids)

    if k==0:
        return 0

    return len(retrieved&gold)/k


def test_retrieval():
    ks=[1,3,5,10]

    recall_totals={k:0 for k in ks}
    precision_totals={k:0 for k in ks}

    count=0

    for claim in claims:
        if "evidence" not in claim:
            continue

        gold_docids=[
            int(docid)
            for docid in claim["evidence"].keys()
        ]

        if not gold_docids:
            continue

        results=retrieve(
            claim["claim"],
            tfidf,
            idf,
            index,
            max(ks)
        )

        for k in ks:
            recall_totals[k]+=recall_at_k(
                results,
                gold_docids,
                k
            )

            precision_totals[k]+=precision_at_k(
                results,
                gold_docids,
                k
            )

        count+=1

    assert count>0

    print()
    print("Retrieval Evaluation")
    print("--------------------")
    print("Evaluated claims:",count)
    print()

    for k in ks:
        recall=recall_totals[k]/count
        precision=precision_totals[k]/count

        print(f"Recall@{k}: {recall:.4f}")
        print(f"Precision@{k}: {precision:.4f}")
        print()


def test_retrieval_results_are_ranked():
    claim=claims[0]

    results=retrieve(
        claim["claim"],
        tfidf,
        idf,
        index,
        10
    )

    assert len(results)<=10

    for i in range(len(results)-1):
        assert results[i][1]>=results[i+1][1]


def test_retrieval_result_format():
    claim=claims[0]

    results=retrieve(
        claim["claim"],
        tfidf,
        idf,
        index,
        5
    )

    for docid,score in results:
        assert isinstance(docid,int)
        assert isinstance(score,float)
        assert score>0