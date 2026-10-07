from src.explain import (
    build_trace,
    format_trace,
    explain_trace
)


def make_pair(
    docid,
    sentence_id,
    label,
    confidence,
    support,
    contradict,
    neutral
):
    return {
        "claim":"The treatment improves survival.",
        "docid":docid,
        "sentence_id":sentence_id,
        "evidence":"The treatment improved survival.",
        "document_score":0.75,
        "evidence_score":4.50,
        "label":label,
        "confidence":confidence,
        "support_probability":support,
        "contradict_probability":contradict,
        "neutral_probability":neutral
    }


def make_verdict():

    return {
        "verdict":"SUPPORT",
        "support_score":0.82,
        "contradict_score":0.08,
        "neutral_score":0.10,
        "support_evidence":[],
        "contradict_evidence":[],
        "conflict":False,
        "reason":"Supporting evidence outweighs contradicting evidence."
    }


def make_verification():

    return {
        "verdict":"SUPPORT",
        "verified":True,
        "confidence":0.82,
        "coverage":0.50,
        "reason":"Verdict passed verification."
    }


def test_build_trace():

    claim="The treatment improves survival."

    document_results=[
        (123,0.75),
        (456,0.61)
    ]

    pairs=[
        make_pair(
            123,
            4,
            "SUPPORT",
            0.90,
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            456,
            2,
            "NEUTRAL",
            0.70,
            0.10,
            0.10,
            0.80
        )
    ]

    trace=build_trace(
        claim,
        document_results,
        pairs,
        make_verdict(),
        make_verification()
    )

    assert trace["claim"]==claim

    assert len(trace["documents"])==2
    assert trace["documents"][0]["rank"]==1
    assert trace["documents"][0]["docid"]==123
    assert trace["documents"][1]["rank"]==2

    assert len(trace["evidence"])==2

    assert trace["evidence"][0]["docid"]==123
    assert trace["evidence"][0]["document_rank"]==1
    assert trace["evidence"][0]["evidence_rank"]==1

    assert trace["evidence"][1]["docid"]==456
    assert trace["evidence"][1]["document_rank"]==2
    assert trace["evidence"][1]["evidence_rank"]==1

    assert trace["final_verdict"]=="SUPPORT"


def test_evidence_ranks_reset_per_document():

    document_results=[
        (123,0.80),
        (456,0.70)
    ]

    pairs=[
        make_pair(
            123,
            1,
            "SUPPORT",
            0.90,
            0.90,
            0.05,
            0.05
        ),
        make_pair(
            123,
            2,
            "SUPPORT",
            0.85,
            0.85,
            0.05,
            0.10
        ),
        make_pair(
            456,
            3,
            "SUPPORT",
            0.80,
            0.80,
            0.05,
            0.15
        )
    ]

    trace=build_trace(
        "test claim",
        document_results,
        pairs,
        make_verdict(),
        make_verification()
    )

    assert trace["evidence"][0]["evidence_rank"]==1
    assert trace["evidence"][1]["evidence_rank"]==2
    assert trace["evidence"][2]["evidence_rank"]==1


def test_format_trace():

    trace=build_trace(
        "The treatment improves survival.",
        [(123,0.75)],
        [
            make_pair(
                123,
                4,
                "SUPPORT",
                0.90,
                0.90,
                0.05,
                0.05
            )
        ],
        make_verdict(),
        make_verification()
    )

    output=format_trace(trace)

    assert isinstance(output,str)

    assert "CLAIM" in output
    assert "DOCUMENTS" in output
    assert "EVIDENCE" in output
    assert "PRELIMINARY VERDICT" in output
    assert "VERIFICATION" in output
    assert "FINAL VERDICT" in output

    assert "Doc 123" in output
    assert "SUPPORT" in output
    assert "0.9000" in output


def test_explain_trace():

    output=explain_trace(
        "The treatment improves survival.",
        [(123,0.75)],
        [
            make_pair(
                123,
                4,
                "SUPPORT",
                0.90,
                0.90,
                0.05,
                0.05
            )
        ],
        make_verdict(),
        make_verification()
    )

    assert isinstance(output,str)
    assert "FINAL VERDICT" in output
    assert output.endswith("SUPPORT")