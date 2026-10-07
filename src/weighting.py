import math


def calculate_idf(index,number_of_documents): #idf is inverted document frequency
    idf={}

    for term,data in index.items():
        idf[term]=math.log((number_of_documents+1)/(data["df"]+1))+1 #+1 to prevent log0 or div by 0 errors

    return idf


def build_tfidf(index,idf): #tfidf is term frequency x inverted document frequency
    tfidf={}

    for term,data in index.items():
        for posting in data["postings"]: #go into postings for term
            docid=posting["docid"] #retrieve a docid
            tf=posting["tf"]

            weight=(1+math.log(tf))*idf[term] #compute tfidf=tf*idf

            if docid not in tfidf:
                tfidf[docid]={} #if tfidf has not been previously computed for the docid, add it

            tfidf[docid][term]=weight #add tfidf of specific term

    return tfidf