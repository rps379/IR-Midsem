import torch

from transformers import AutoTokenizer,AutoModelForSequenceClassification


MODEL_NAME="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLIModel:
    def __init__(
        self,
        model_name=MODEL_NAME,
        batch_size=16
    ):
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name
        )

        self.model=AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

        self.batch_size=batch_size

    def classify(self,claim,evidence):
        results=self.predict_batch([
            {
                "claim":claim,
                "evidence":evidence
            }
        ])

        return results[0]

    def predict_batch(self,pairs):
        if not pairs:
            return []

        results=[]

        for start in range(
            0,
            len(pairs),
            self.batch_size
        ):
            batch=pairs[
                start:start+self.batch_size
            ]

            claims=[]
            evidences=[]

            for pair in batch:
                claims.append(
                    pair["claim"]
                )

                evidences.append(
                    pair["evidence"]
                )

            inputs=self.tokenizer(
                evidences,
                claims,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs=self.model(**inputs)

            probabilities=torch.softmax(
                outputs.logits,
                dim=-1
            )

            for i in range(len(batch)):
                support_probability=float(
                    probabilities[i][0]
                )

                neutral_probability=float(
                    probabilities[i][1]
                )

                contradict_probability=float(
                    probabilities[i][2]
                )

                label_index=int(
                    torch.argmax(
                        probabilities[i]
                    )
                )

                if label_index==0:
                    label="SUPPORT"

                elif label_index==1:
                    label="NEUTRAL"

                else:
                    label="CONTRADICT"

                confidence=float(
                    probabilities[i][label_index]
                )

                results.append({
                    "label":label,
                    "confidence":confidence,
                    "contradict_probability":
                        contradict_probability,
                    "support_probability":
                        support_probability,
                    "neutral_probability":
                        neutral_probability
                })

        return results

    def predict(self,pairs):
        return self.predict_batch(pairs)


def classify_pairs(pairs,model=None):
    if not pairs:
        return []

    if model is None:
        model=NLIModel()

    if hasattr(model,"predict_batch"):
        predictions=model.predict_batch(
            pairs
        )

        results=[]

        for i,pair in enumerate(pairs):
            result=dict(pair)

            result.update(
                predictions[i]
            )

            results.append(result)

        return results

    results=[]

    for pair in pairs:
        result=dict(pair)

        classification=model.classify(
            pair["claim"],
            pair["evidence"]
        )

        result.update(
            classification
        )

        results.append(result)

    return results