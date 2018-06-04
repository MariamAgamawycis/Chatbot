import random
from flask import Flask, request
from pymessenger.bot import Bot
import os
import string
import sys
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
ACCESS_TOKEN = 'EAADBkvb3F4wBALo3ZBAJmnFEqaq3JkG9Ruuxt0psgvhZAK5BPksdQHL3MOZAwehzaIrFwb13HZBmg31rLmVyx2CpAgRZARjXa2GWXAPYt58eZASZBfzHKii2NYjusBMvys2isNtWyOWu3zr2NxDotBgcZAMpD79BN2xG0Goui2zEiwZDZD'
VERIFY_TOKEN = 'VERIFY_TOKEN'
bot = Bot (ACCESS_TOKEN)

@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    else:
       output = request.get_json()
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('message'):
                recipient_id = message['sender']['id']
                if message['message'].get('text'):
                    response_sent_text = get_message(message['message'].get('text'))
                    send_message(recipient_id, response_sent_text)
                if message['message'].get('attachments'):
                    response_sent_nontext = get_message(message['message'].get('attachments'))
                    send_message(recipient_id, response_sent_nontext)
    return "Message Processed"


def verify_fb_token(token_sent):
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'

def get_message(msg):
    if Classify(msg) == 'hi':
        sample_responses = ["Hi!", "Nice to see you", "Hey dear", "We're greatful to know you :)"]
        return random.choice(sample_responses)
    elif Classify(msg) == 'weather':
        sample_responses = ["it's cloudy today", "it's rainy today", "it's cold today", "it's sunny today"]
        return random.choice(sample_responses)

def send_message(recipient_id, response):
    bot.send_text_message(recipient_id, response)
    return "success"

def Classify(text):
    data = pd.read_csv("data.txt")
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

    docs_new = text
    docs = [docs_new]
    X_new_counts = count_vect.transform(docs)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
    return predicted


if __name__ == "__main__":
    app.run()