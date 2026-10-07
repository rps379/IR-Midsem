import json

#this reads the corpus
def read_corpus(filename):
    corpus={}

    with open(filename,"r",encoding="utf-8") as file:
        for line in file:
            doc=json.loads(line)#loads a line in the json
            #corpus contains doc_id(id of the document)
            #title(title of the document)
            #abstract(abstract of the scientific study with sentences in a list)
            #structured(boolean value whether the sections are specifically subheaded)
            corpus[doc["doc_id"]]={
                "title":doc["title"],
                "abstract":doc["abstract"],
                "structured":doc["structured"]
            }

    return corpus

def read_claims(filename):
    claims=[]

    with open(filename,"r",encoding="utf-8") as file:
        for line in file:
            claims.append(json.loads(line))

    return claims