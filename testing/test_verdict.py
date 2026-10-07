from src.verdict import verdict_nli


def make_pair(
    label,
    confidence,
    support,
    contradict,
    neutral
):
    return {
        "claim":"test claim",
        "docid":1,
        "sentence_id":0,
        "evidence":"test evidence",
        "document_score":0.5,
        "evidence_score":1.0,
        "label":label,
        "confidence":confidence,
        "support_probability":support,
        "contradict_probability":contradict,
        "neutral_probability":neutral
    }


def test_support_aggregation():

    pairs=[
        make_pair(
            "SUPPORT",
            0.90,
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "SUPPORT",
            0.85,
            0.85,
            0.05,
            0.10
        )
    ]

    result=verdict_nli(pairs)

    assert result["verdict"]=="SUPPORT"
    assert result["conflict"] is False
    assert result["support_score"]>result["contradict_score"]


def test_contradict_aggregation():

    pairs=[
        make_pair(
            "CONTRADICT",
            0.90,
            0.05,
            0.90,
            0.05
        ),
        make_pair(
            "CONTRADICT",
            0.85,
            0.05,
            0.85,
            0.10
        )
    ]

    result=verdict_nli(pairs)

    assert result["verdict"]=="CONTRADICT"
    assert result["conflict"] is False
    assert result["contradict_score"]>result["support_score"]


def test_neutral_aggregation_abstains():

    pairs=[
        make_pair(
            "NEUTRAL",
            0.70,
            0.10,
            0.10,
            0.80
        ),
        make_pair(
            "NEUTRAL",
            0.70,
            0.15,
            0.10,
            0.75
        )
    ]

    result=verdict_nli(pairs)

    assert result["verdict"]=="ABSTAIN"
    assert result["conflict"] is False


def test_conflicting_evidence_abstains():

    pairs=[
        make_pair(
            "SUPPORT",
            0.90,
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "CONTRADICT",
            0.90,
            0.05,
            0.90,
            0.05
        )
    ]

    result=verdict_nli(pairs)

    assert result["conflict"] is True
    assert result["verdict"]=="ABSTAIN"


def test_empty_input_abstains():

    result=verdict_nli([])

    assert result["verdict"]=="ABSTAIN"
    assert result["support_score"]==0.0
    assert result["contradict_score"]==0.0
    assert result["neutral_score"]==0.0
    assert result["conflict"] is False


def test_support_evidence_is_preserved():

    pair=make_pair(
        "SUPPORT",
        0.92,
        0.92,
        0.03,
        0.05
    )

    result=verdict_nli([pair])

    assert len(result["support_evidence"])==1
    assert result["support_evidence"][0] is pair
    assert len(result["contradict_evidence"])==0