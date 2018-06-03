import string
import sys
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

def classification(xinput):
    data = pd.read_csv("F:\Chatbot\data.txt")
    input = data["message"]
    output = data["intent"]

    stopWords = set(stopwords.words('english'))
    stops = list(string.punctuation)
    stops += stopWords

    new_input = []
    for line in input:
        new_word = ""
        for word in line.split():
            if word not in stops:
                new_word += word + " "
        new_input.append(new_word)

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(new_input)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    clf = MultinomialNB()
    clf.fit(X_train_tfidf, output)

    docs_new = xinput
    docs = [docs_new]
    X_new_counts = count_vect.transform(docs)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
    return(predicted)
