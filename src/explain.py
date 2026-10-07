def build_trace(
    claim,
    document_results,
    classified_pairs,
    verdict_result,
    verification_result
):
    documents=[]

    for rank,(docid,score) in enumerate(
        document_results,
        start=1
    ):
        documents.append({
            "rank":rank,
            "docid":docid,
            "score":score
        })

    evidence=[]

    document_ranks={}

    for rank,(docid,score) in enumerate(
        document_results,
        start=1
    ):
        document_ranks[docid]=rank

    for rank,pair in enumerate(
        classified_pairs,
        start=1
    ):
        docid=pair["docid"]

        evidence.append({
            "docid":docid,
            "document_rank":document_ranks.get(
                docid
            ),
            "sentence_id":pair["sentence_id"],
            "evidence_rank":rank,
            "evidence":pair["evidence"],
            "document_score":pair["document_score"],
            "evidence_score":pair["evidence_score"],
            "bm25_score":pair.get(
                "bm25_score",
                pair["evidence_score"]
            ),
            "reranker_score":pair.get(
                "reranker_score"
            ),
            "nli_relevance_score":pair.get(
                "nli_relevance_score"
            ),
            "label":pair["label"],
            "confidence":pair["confidence"],
            "support_probability":pair[
                "support_probability"
            ],
            "contradict_probability":pair[
                "contradict_probability"
            ],
            "neutral_probability":pair[
                "neutral_probability"
            ]
        })

    return {
        "claim":claim,
        "documents":documents,
        "evidence":evidence,
        "preliminary_verdict":verdict_result,
        "verification":verification_result,
        "final_verdict":verification_result[
            "verdict"
        ]
    }


def format_trace(trace):
    lines=[]

    lines.append("CLAIM")
    lines.append("-----")
    lines.append(trace["claim"])
    lines.append("")

    lines.append("DOCUMENTS")
    lines.append("---------")

    for document in trace["documents"]:
        lines.append(
            f'{document["rank"]}. '
            f'Doc {document["docid"]} '
            f'score={document["score"]:.4f}'
        )

    lines.append("")

    lines.append("EVIDENCE")
    lines.append("--------")

    for evidence in trace["evidence"]:
        lines.append(
            f'{evidence["evidence_rank"]}. '
            f'Doc {evidence["docid"]} '
            f'/ sentence {evidence["sentence_id"]}'
        )

        lines.append(
            f'  Document rank: '
            f'{evidence["document_rank"]}'
        )

        lines.append(
            f'  Evidence rank: '
            f'{evidence["evidence_rank"]}'
        )

        lines.append(
            f'  BM25 score: '
            f'{evidence["bm25_score"]:.4f}'
        )

        if evidence["reranker_score"] is not None:
            lines.append(
                f'  Cross-encoder score: '
                f'{evidence["reranker_score"]:.4f}'
            )

        if evidence["nli_relevance_score"] is not None:
            lines.append(
                f'  NLI relevance score: '
                f'{evidence["nli_relevance_score"]:.4f}'
            )

        lines.append(
            f'  NLI: '
            f'{evidence["label"]} '
            f'confidence='
            f'{evidence["confidence"]:.4f}'
        )

        lines.append(
            f'  SUPPORT='
            f'{evidence["support_probability"]:.4f} '
            f'CONTRADICT='
            f'{evidence["contradict_probability"]:.4f} '
            f'NEUTRAL='
            f'{evidence["neutral_probability"]:.4f}'
        )

        lines.append(
            f'  "{evidence["evidence"]}"'
        )

        lines.append("")

    preliminary=trace["preliminary_verdict"]

    lines.append("PRELIMINARY VERDICT")
    lines.append("-------------------")

    lines.append(
        f'Verdict: {preliminary["verdict"]}'
    )

    lines.append(
        f'Support score: '
        f'{preliminary["support_score"]:.4f}'
    )

    lines.append(
        f'Contradict score: '
        f'{preliminary["contradict_score"]:.4f}'
    )

    lines.append(
        f'Neutral score: '
        f'{preliminary["neutral_score"]:.4f}'
    )

    lines.append(
        f'Conflict: '
        f'{preliminary["conflict"]}'
    )

    if "reason" in preliminary:
        lines.append(
            f'Reason: {preliminary["reason"]}'
        )

    lines.append("")

    verification=trace["verification"]

    lines.append("VERIFICATION")
    lines.append("------------")

    lines.append(
        f'Verified: '
        f'{verification["verified"]}'
    )

    lines.append(
        f'Confidence: '
        f'{verification["confidence"]:.4f}'
    )

    lines.append(
        f'Coverage: '
        f'{verification["coverage"]:.4f}'
    )

    lines.append(
        f'Reason: '
        f'{verification["reason"]}'
    )

    lines.append("")

    lines.append("FINAL VERDICT")
    lines.append("-------------")

    lines.append(
        trace["final_verdict"]
    )

    return "\n".join(lines)


def explain_trace(
    claim,
    document_results,
    classified_pairs,
    verdict_result,
    verification_result
):
    trace=build_trace(
        claim,
        document_results,
        classified_pairs,
        verdict_result,
        verification_result
    )

    return format_trace(trace)