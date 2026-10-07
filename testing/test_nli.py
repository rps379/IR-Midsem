from src.nli import NLIModel,classify_pairs


class FakeNLIModel:

    def predict_batch(self,pairs):

        results=[]

        for pair in pairs:
            results.append({
                "label":"SUPPORT",
                "confidence":0.9,
                "contradict_probability":0.05,
                "support_probability":0.9,
                "neutral_probability":0.05
            })

        return results


def test_nli_prediction_structure():

    model=FakeNLIModel()

    result=model.predict_batch([
        {
            "claim":"The treatment improves survival.",
            "evidence":"The treatment was associated with improved survival."
        }
    ])

    assert len(result)==1

    prediction=result[0]

    assert prediction["label"] in {
        "SUPPORT",
        "CONTRADICT",
        "NEUTRAL"
    }

    assert isinstance(
        prediction["confidence"],
        float
    )

    assert 0<=prediction["confidence"]<=1

    assert 0<=prediction["contradict_probability"]<=1
    assert 0<=prediction["support_probability"]<=1
    assert 0<=prediction["neutral_probability"]<=1


def test_nli_batch_size():

    model=FakeNLIModel()

    pairs=[
        {
            "claim":"Claim "+str(i),
            "evidence":"Evidence "+str(i)
        }
        for i in range(20)
    ]

    results=model.predict_batch(pairs)

    assert len(results)==20


def test_classify_pairs_preserves_pair_information():

    pairs=[
        {
            "claim":"The treatment reduces mortality.",
            "docid":123,
            "sentence_id":4,
            "evidence":"The treatment reduced mortality.",
            "document_score":0.72,
            "evidence_score":4.21
        }
    ]

    results=classify_pairs(
        pairs,
        model=FakeNLIModel()
    )

    assert len(results)==1

    result=results[0]

    assert result["claim"]==pairs[0]["claim"]
    assert result["docid"]==123
    assert result["sentence_id"]==4
    assert result["evidence"]==pairs[0]["evidence"]

    assert result["document_score"]==0.72
    assert result["evidence_score"]==4.21

    assert result["label"]=="SUPPORT"
    assert result["confidence"]==0.9


def test_empty_pairs():

    results=classify_pairs(
        [],
        model=FakeNLIModel()
    )

    assert results==[]


def test_real_nli_model():

    import os

    if os.environ.get("RUN_NLI_MODEL_TEST")!="1":
        return

    model=NLIModel(
        batch_size=2
    )

    results=model.predict_batch([
        {
            "claim":"The treatment improves survival.",
            "evidence":"The treatment was associated with improved survival."
        },
        {
            "claim":"The treatment improves survival.",
            "evidence":"The treatment was associated with increased mortality."
        }
    ])

    assert len(results)==2

    for result in results:

        assert result["label"] in {
            "SUPPORT",
            "CONTRADICT",
            "NEUTRAL"
        }

        assert 0<=result["confidence"]<=1

        total=(
            result["contradict_probability"]+
            result["support_probability"]+
            result["neutral_probability"]
        )

        assert abs(total-1.0)<0.001