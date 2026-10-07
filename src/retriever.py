import heapq
import math

from src.preprocess import preprocess


def build_query_vector(query,idf):  #building query vector
    tokens=preprocess(query)  #preprocess the query

    frequencies={}

    for token in tokens:
        if token not in frequencies:
            frequencies[token]=0

        frequencies[token]+=1  #calculate freq per token

    vector={}

    for term,tf in frequencies.items():
        if term in idf:
            vector[term]=(1+math.log(tf))*idf[term]  #find query weight(idf is a dictionary containing idf of all terms)

    return vector


def cosine_similarity(query_vector,document_vector): #computing cosine similarity
    dot_product=0
    query_length=0
    document_length=0

    for term,weight in query_vector.items():
        query_length+=weight*weight  #length=w*w(for query)

        if term in document_vector:
            dot_product+=weight*document_vector[term] #if the term is in a document, dot product=query_wt*document_wt

    for weight in document_vector.values():
        document_length+=weight*weight #length=w*w(for document)

    if query_length==0 or document_length==0:
        return 0

    return dot_product/(math.sqrt(query_length)*math.sqrt(document_length)) #final dot product=summation(query_wt*document_wt)/
                                                                            #root(query_length)*root(doc_length)


def get_candidate_documents(query_vector,index): #get documents that were in each terms postings
    candidates=set()

    for term in query_vector:
        if term not in index:
            continue

        for posting in index[term]["postings"]:
            candidates.add(posting["docid"])

    return candidates


def retrieve(query,tfidf,idf,index,k=10):
    query_vector=build_query_vector(query,idf) #build the query vector for given query

    if not query_vector: #if no query vector returned, do not proceed further
        return []

    candidate_documents=get_candidate_documents(  #get the documents in each terms postings
        query_vector,
        index
    )

    heap=[] #create a heap, we will use a minheap so min score stays at [0][0]

    for docid in candidate_documents:
        if docid not in tfidf: #if docid does not have computed tfidf, continue
            continue

        document_vector=tfidf[docid] 

        score=cosine_similarity( #compute cosine similarity of query vs document with id=docid
            query_vector,
            document_vector
        )

        if score<=0: #if -ve score, continue
            continue

        if len(heap)<k: #if less than 10 elements in heap, keep pushing
            heapq.heappush(
                heap,
                (score,docid)
            )
        elif score>heap[0][0]: #if the current score, is higher than the score at [0][0](smallest score), replace it
            heapq.heapreplace(
                heap,
                (score,docid)
            )

    results=[]

    while heap: #pop all the results and append to results array
        score,docid=heapq.heappop(heap)

        results.append(
            (docid,score)
        )

    results.reverse()

    return results