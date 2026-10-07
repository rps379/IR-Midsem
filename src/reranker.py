import torch

from transformers import AutoTokenizer,AutoModelForSequenceClassification


MODEL_NAME="cross-encoder/ms-marco-MiniLM-L6-v2"


class EvidenceReranker:
    def __init__(self,model_name=MODEL_NAME):
        self.tokenizer=AutoTokenizer.from_pretrained(
            model_name
        )

        self.model=AutoModelForSequenceClassification.from_pretrained(
            model_name
        )

        self.model.eval()

    def rerank(self,claim,evidence_candidates,k=5):
        if not evidence_candidates:
            return []

        claims=[]
        evidences=[]

        for candidate in evidence_candidates:
            claims.append(claim)
            evidences.append(candidate["evidence"])

        inputs=self.tokenizer(
            claims,
            evidences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs=self.model(**inputs)

        scores=outputs.logits.squeeze(-1).tolist()

        if isinstance(scores,float):
            scores=[scores]

        results=[]

        for i,candidate in enumerate(
            evidence_candidates
        ):
            result=dict(candidate)

            result["reranker_score"]=scores[i]

            results.append(result)

        results.sort(
            key=lambda result:(
                -result["reranker_score"],
                result["docid"],
                result["sentence_id"]
            )
        )

        return results[:k]