import re
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#preprocessing of data and queries
lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))

scientific_tokens={"a","b","c","d","e","k"}#certain takens we do not want removed by stopwords
#say "vitamin d", normal stopword application would remove the d, making a doc containing vitamin d vs vitamin e appear the same


def preprocess(text):
    text=text.lower()#convert all to lowercase

    text=re.sub(r"[^a-z0-9\s]","",text)

    tokens=word_tokenize(text)#tokenization

    tokens=[word for word in tokens if word not in stop_words or word in scientific_tokens]

    tokens=[lemmatizer.lemmatize(word) for word in tokens]#lemmization
    #here lemmization is used as in a scientific paper, precise meaning is critical

    return tokens