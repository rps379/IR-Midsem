import json
from pathlib import Path
from collections import Counter

#initial code to inspect the databse, return data about various corpuses, only for testing purposes
DATA_DIR=Path(__file__).parent.parent/"data"


def load(path):
    with open(path,"r",encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def inspect_corpus():
    data=load(DATA_DIR/"corpus.jsonl")

    sentences=[len(x["abstract"]) for x in data]

    print("===== CORPUS =====")
    print("Documents:",len(data))
    print("Unique IDs:",len(set(x["doc_id"] for x in data)))
    print("Total sentences:",sum(sentences))
    print("Average sentences/document:",round(sum(sentences)/len(sentences),2))
    print("Minimum sentences/document:",min(sentences))
    print("Maximum sentences/document:",max(sentences))


def inspect_claims(name):
    data=load(DATA_DIR/f"claims_{name}.jsonl")

    print(f"\n===== {name.upper()} =====")
    print("Claims:",len(data))
    print("Unique IDs:",len(set(x["id"] for x in data)))

    if name=="test":
        return

    labels=Counter()
    evidence_docs=0
    evidence_sentences=0
    cited_docs=0
    claims_with_evidence=0

    for claim in data:
        evidence=claim["evidence"]

        if evidence:
            claims_with_evidence+=1

        evidence_docs+=len(evidence)
        cited_docs+=len(claim["cited_doc_ids"])

        for rationales in evidence.values():
            for rationale in rationales:
                labels[rationale["label"]]+=1
                evidence_sentences+=len(rationale["sentences"])

    print("Claims with evidence:",claims_with_evidence)
    print("Claims without evidence:",len(data)-claims_with_evidence)
    print("Evidence labels:",dict(labels))
    print("Average evidence documents/claim:",round(evidence_docs/len(data),2))
    print("Average evidence sentences/claim:",round(evidence_sentences/len(data),2))
    print("Average cited documents/claim:",round(cited_docs/len(data),2))


def main():
    inspect_corpus()
    inspect_claims("train")
    inspect_claims("dev")
    inspect_claims("test")


if __name__=="__main__":
    main()