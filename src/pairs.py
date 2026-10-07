from src.evidence import retrieve_evidence


def build_claim_evidence_pairs(
    claim,
    document_results,
    corpus,
    idf,
    k=5,
    reranker=None,
    nli_model=None
):
    all_candidates=[]

    for docid,document_score in document_results:
        document=corpus[docid]

        evidence_candidates=retrieve_evidence(
            claim,
            document,
            idf,
            k=15
        )

        for candidate in evidence_candidates:
            result={
                "docid":docid,
                "document_score":document_score,
                "sentence_id":candidate["sentence_id"],
                "evidence":candidate["text"],
                "evidence_score":candidate["score"],
                "bm25_score":candidate["bm25_score"]
            }

            all_candidates.append(result)

    if not all_candidates:
        return []

    if reranker is not None:
        all_candidates=reranker.rerank(
            claim,
            all_candidates,
            k=20
        )
    else:
        all_candidates.sort(
            key=lambda candidate:(
                -candidate["bm25_score"],
                candidate["docid"],
                candidate["sentence_id"]
            )
        )

        all_candidates=all_candidates[:20]

    if nli_model is not None:
        classified=[]

        for candidate in all_candidates:
            result=nli_model.classify(
                claim,
                candidate["evidence"]
            )

            candidate=dict(candidate)

            candidate["label"]=result["label"]
            candidate["confidence"]=result["confidence"]
            candidate["support_probability"]=result[
                "support_probability"
            ]
            candidate["contradict_probability"]=result[
                "contradict_probability"
            ]
            candidate["neutral_probability"]=result[
                "neutral_probability"
            ]

            classified.append(candidate)

        all_candidates=classified

    return all_candidates[:k]