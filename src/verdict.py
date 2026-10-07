def verdict_nli(
    classified_pairs,
    support_threshold=0.70,
    contradict_threshold=0.70,
    conflict_margin=0.10
):
    if not classified_pairs:
        return {
            "verdict":"ABSTAIN",
            "support_score":0.0,
            "contradict_score":0.0,
            "neutral_score":0.0,
            "support_evidence":[],
            "contradict_evidence":[],
            "conflict":False
        }

    support_evidence=[]
    contradict_evidence=[]

    support_score=0.0
    contradict_score=0.0
    neutral_score=0.0

    for pair in classified_pairs:
        support_probability=pair["support_probability"]
        contradict_probability=pair["contradict_probability"]
        neutral_probability=pair["neutral_probability"]

        support_score+=support_probability
        contradict_score+=contradict_probability
        neutral_score+=neutral_probability

        if (
            pair["label"]=="SUPPORT"
            and
            pair["confidence"]>=support_threshold
        ):
            support_evidence.append(pair)

        if (
            pair["label"]=="CONTRADICT"
            and
            pair["confidence"]>=contradict_threshold
        ):
            contradict_evidence.append(pair)

    count=len(classified_pairs)

    support_score/=count
    contradict_score/=count
    neutral_score/=count

    strong_support=bool(support_evidence)
    strong_contradict=bool(contradict_evidence)

    conflict=(
        strong_support
        and
        strong_contradict
        and
        abs(
            support_score-
            contradict_score
        )<=conflict_margin
    )

    if conflict:
        verdict="ABSTAIN"

    elif (
        strong_support
        and
        support_score>contradict_score
        and
        support_score>neutral_score
    ):
        verdict="SUPPORT"

    elif (
        strong_contradict
        and
        contradict_score>support_score
        and
        contradict_score>neutral_score
    ):
        verdict="CONTRADICT"

    else:
        verdict="ABSTAIN"

    return {
        "verdict":verdict,
        "support_score":support_score,
        "contradict_score":contradict_score,
        "neutral_score":neutral_score,
        "support_evidence":support_evidence,
        "contradict_evidence":contradict_evidence,
        "conflict":conflict
    }