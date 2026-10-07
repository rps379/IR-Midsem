from src.preprocess import preprocess


def build_inverted_index(corpus):
    index={}

    for docid,document in corpus.items():
        text=" ".join(document["abstract"])
        tokens=preprocess(text)

        frequencies={}

        for token in tokens:
            if token not in frequencies:
                frequencies[token]=0

            frequencies[token]+=1

        for term,tf in frequencies.items():
            if term not in index:
                index[term]={
                    "df":0,
                    "postings":[]
                }

            index[term]["df"]+=1

            index[term]["postings"].append({
                "docid":docid,
                "tf":tf
            })

    return index