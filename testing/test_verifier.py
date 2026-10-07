from src.verifier import (
    calculate_evidence_coverage,
    calculate_confidence,
    verify_verdict
)


def make_pair(
    label,
    support,
    contradict,
    neutral
):
    return {
        "claim":"test claim",
        "docid":1,
        "sentence_id":0,
        "evidence":"test evidence",
        "document_score":0.8,
        "evidence_score":1.0,
        "label":label,
        "confidence":max(
            support,
            contradict,
            neutral
        ),
        "support_probability":support,
        "contradict_probability":contradict,
        "neutral_probability":neutral
    }


def test_support_verification():

    pairs=[
        make_pair(
            "SUPPORT",
            0.95,
            0.03,
            0.02
        ),
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "NEUTRAL",
            0.10,
            0.10,
            0.80
        )
    ]

    verdict_result={
        "verdict":"SUPPORT",
        "conflict":False
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="SUPPORT"
    assert result["verified"] is True
    assert result["confidence"]>0.70
    assert result["coverage"]>=0.40


def test_contradict_verification():

    pairs=[
        make_pair(
            "CONTRADICT",
            0.03,
            0.95,
            0.02
        ),
        make_pair(
            "CONTRADICT",
            0.05,
            0.90,
            0.05
        ),
        make_pair(
            "NEUTRAL",
            0.10,
            0.10,
            0.80
        )
    ]

    verdict_result={
        "verdict":"CONTRADICT",
        "conflict":False
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="CONTRADICT"
    assert result["verified"] is True


def test_insufficient_evidence_abstains():

    pairs=[
        make_pair(
            "SUPPORT",
            0.95,
            0.03,
            0.02
        )
    ]

    verdict_result={
        "verdict":"SUPPORT",
        "conflict":False
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="ABSTAIN"
    assert result["verified"] is False


def test_low_confidence_abstains():

    pairs=[
        make_pair(
            "SUPPORT",
            0.60,
            0.20,
            0.20
        ),
        make_pair(
            "SUPPORT",
            0.65,
            0.20,
            0.15
        )
    ]

    verdict_result={
        "verdict":"SUPPORT",
        "conflict":False
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="ABSTAIN"
    assert result["verified"] is False


def test_conflict_abstains():

    pairs=[
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "CONTRADICT",
            0.05,
            0.90,
            0.05
        )
    ]

    verdict_result={
        "verdict":"ABSTAIN",
        "conflict":True
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="ABSTAIN"
    assert result["verified"] is False


def test_abstained_verdict_stays_abstained():

    pairs=[
        make_pair(
            "SUPPORT",
            0.95,
            0.03,
            0.02
        ),
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        )
    ]

    verdict_result={
        "verdict":"ABSTAIN",
        "conflict":False
    }

    result=verify_verdict(
        pairs,
        verdict_result
    )

    assert result["verdict"]=="ABSTAIN"
    assert result["verified"] is False


def test_evidence_coverage():

    pairs=[
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "NEUTRAL",
            0.10,
            0.10,
            0.80
        ),
        make_pair(
            "NEUTRAL",
            0.10,
            0.10,
            0.80
        )
    ]

    coverage=calculate_evidence_coverage(
        pairs,
        "SUPPORT"
    )

    assert coverage==0.5


def test_confidence():

    pairs=[
        make_pair(
            "SUPPORT",
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            "SUPPORT",
            0.80,
            0.10,
            0.10
        )
    ]

    confidence=calculate_confidence(
        pairs,
        "SUPPORT"
    )

    assert abs(confidence-0.85)<1e-9