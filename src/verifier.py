def calculate_evidence_coverage(
    classified_pairs,
    verdict
):
    if not classified_pairs:
        return 0.0

    relevant_pairs=[]

    for pair in classified_pairs:
        if verdict=="SUPPORT":
            if pair["label"]=="SUPPORT":
                relevant_pairs.append(pair)

        elif verdict=="CONTRADICT":
            if pair["label"]=="CONTRADICT":
                relevant_pairs.append(pair)

    return len(relevant_pairs)/len(classified_pairs)


def calculate_confidence(
    classified_pairs,
    verdict
):
    probabilities=[]

    for pair in classified_pairs:
        if verdict=="SUPPORT":
            if pair["label"]=="SUPPORT":
                probabilities.append(
                    pair["support_probability"]
                )

        elif verdict=="CONTRADICT":
            if pair["label"]=="CONTRADICT":
                probabilities.append(
                    pair["contradict_probability"]
                )

    if not probabilities:
        return 0.0

    return sum(probabilities)/len(probabilities)


def verify_verdict(
    classified_pairs,
    verdict_result,
    minimum_evidence=2,
    minimum_confidence=0.70,
    minimum_coverage=0.40
):
    verdict=verdict_result["verdict"]

    if verdict=="ABSTAIN":
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":0.0,
            "coverage":0.0,
            "reason":"The verdict was already abstained."
        }

    if not classified_pairs:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":0.0,
            "coverage":0.0,
            "reason":"No evidence was available."
        }

    if verdict=="SUPPORT":
        relevant_evidence=[
            pair
            for pair in classified_pairs
            if (
                pair["label"]=="SUPPORT"
                and
                pair["support_probability"]>=minimum_confidence
            )
        ]

    elif verdict=="CONTRADICT":
        relevant_evidence=[
            pair
            for pair in classified_pairs
            if (
                pair["label"]=="CONTRADICT"
                and
                pair["contradict_probability"]>=minimum_confidence
            )
        ]

    else:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":0.0,
            "coverage":0.0,
            "reason":"Unknown verdict."
        }

    confidence=calculate_confidence(
        classified_pairs,
        verdict
    )

    coverage=calculate_evidence_coverage(
        classified_pairs,
        verdict
    )

    if verdict_result["conflict"]:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":confidence,
            "coverage":coverage,
            "reason":"Supporting and contradicting evidence conflict."
        }

    if len(relevant_evidence)<minimum_evidence:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":confidence,
            "coverage":coverage,
            "reason":"Insufficient high-confidence evidence."
        }

    if confidence<minimum_confidence:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":confidence,
            "coverage":coverage,
            "reason":"Evidence confidence is below the verification threshold."
        }

    if coverage<minimum_coverage:
        return {
            "verdict":"ABSTAIN",
            "verified":False,
            "confidence":confidence,
            "coverage":coverage,
            "reason":"Evidence coverage is too low."
        }

    return {
        "verdict":verdict,
        "verified":True,
        "confidence":confidence,
        "coverage":coverage,
        "reason":"Verdict passed verification."
    }