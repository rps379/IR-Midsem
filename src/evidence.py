import heapq
import math

from src.preprocess import preprocess


def build_sentence_vectors(document,idf):
    sentence_vectors=[]

    for sentence_id,sentence in enumerate(document["abstract"]):
        tokens=preprocess(sentence)

        frequencies={}

        for token in tokens:
            if token not in frequencies:
                frequencies[token]=0

            frequencies[token]+=1

        sentence_vectors.append({
            "sentence_id":sentence_id,
            "text":sentence,
            "tokens":tokens,
            "frequencies":frequencies,
            "length":len(tokens)
        })

    return sentence_vectors


def build_query_terms(claim,idf):
    tokens=preprocess(claim)

    frequencies={}

    for token in tokens:
        if token not in frequencies:
            frequencies[token]=0

        frequencies[token]+=1

    return {
        term:tf
        for term,tf in frequencies.items()
        if term in idf
    }


def calculate_average_sentence_length(sentence_vectors):
    if not sentence_vectors:
        return 0

    total_length=0

    for sentence in sentence_vectors:
        total_length+=sentence["length"]

    return total_length/len(sentence_vectors)


def bm25_score(
    query_terms,
    sentence,
    idf,
    average_sentence_length,
    k1=1.5,
    b=0.75
):
    if sentence["length"]==0:
        return 0

    score=0

    length_normalization=(
        1-b+
        b*sentence["length"]/average_sentence_length
    )

    for term in query_terms:
        if term not in sentence["frequencies"]:
            continue

        tf=sentence["frequencies"][term]

        numerator=tf*(k1+1)

        denominator=(
            tf+
            k1*length_normalization
        )

        score+=idf[term]*(numerator/denominator)

    return score


def retrieve_evidence(claim,document,idf,k=5):
    query_terms=build_query_terms(
        claim,
        idf
    )

    if not query_terms:
        return []

    sentence_vectors=build_sentence_vectors(
        document,
        idf
    )

    average_sentence_length=calculate_average_sentence_length(
        sentence_vectors
    )

    if average_sentence_length==0:
        return []

    candidate_k=15

    heap=[]

    for sentence in sentence_vectors:
        score=bm25_score(
            query_terms,
            sentence,
            idf,
            average_sentence_length
        )

        if score<=0:
            continue

        sentence_id=sentence["sentence_id"]

        result={
            "sentence_id":sentence_id,
            "text":sentence["text"],
            "score":score,
            "bm25_score":score
        }

        if len(heap)<candidate_k:
            heapq.heappush(
                heap,
                (score,sentence_id,result)
            )

        elif score>heap[0][0]:
            heapq.heapreplace(
                heap,
                (score,sentence_id,result)
            )

    results=[]

    while heap:
        score,sentence_id,result=heapq.heappop(heap)
        results.append(result)

    results.reverse()

    return results[:k]