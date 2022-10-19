import math

import nltk
import pandas as pd
from nltk.corpus import stopwords
import string
import re

from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline


def cleaning_stopwords(text, stop_words):
    return " ".join([word for word in str(text).split() if word not in stop_words])


def cleaning_punctuations(text):
    punctuations_list = string.punctuation
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


def tokenization(data):
    return data.split()


def lemmatizer_on_text(data, lemmatizer):
    text = [lemmatizer.lemmatize(word) for word in data]
    return data


def calculate_mle_for_tweet(language_model, data):
    mle_score = 1
    for word in data:
        mle_score += language_model.score(word)
    return round(mle_score, 2)

def preprocess(data):
    # Remove mentions (@USER)
    data.loc[:, 'tweet'] = data.tweet.str.replace('@USER', '')
    # Converting the text to lowercase
    data['tweet'] = data['tweet'].str.lower()
    # data['tweet'].tail()

    # Remove stop words from each row
    stop_words = set(stopwords.words('english'))
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_stopwords(text, stop_words))

    # Removing all the punctuations
    data['tweet'] = data['tweet'].apply(lambda x: cleaning_punctuations(x))

    # Cleaning numbers
    data['tweet'] = data['tweet'].apply(lambda x: cleaning_numbers(x))

    # Removing emojis
    data.loc[:, 'tweet'] = data.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )

    # Tokenization
    data['tweet'] = data['tweet'].apply(lambda x: tokenization(x))

    # Lemmatizing the text
    lemmatizer = nltk.WordNetLemmatizer()
    data['tweet'] = data['tweet'].apply(lambda x: lemmatizer_on_text(x, lemmatizer))

    return data


def train_LM(train_file_path):
    train_data = pd.read_csv(train_file_path, sep='\t')

    train_data = preprocess(train_data)
    train, vocab = padded_everygram_pipeline(2, train_data['tweet'])
    # print(train)
    lm_full = MLE(2)
    # print(lm.vocab)
    lm_full.fit(train, vocab)
    # print(lm.vocab)
    # print(lm.vocab.lookup(["americans", "this"]))

    # Non-offensive
    non_offensive = train_data.loc[train_data['subtask_a'] == "NOT"]
    print(non_offensive.head())
    non_offensive_train, non_offensive_vocab = padded_everygram_pipeline(2, non_offensive['tweet'])
    lm_not = MLE(2)
    lm_not.fit(non_offensive_train, non_offensive_vocab)

    # Offensive
    offensive = train_data.loc[train_data['subtask_a'] == "OFF"]
    print(offensive.head())
    offensive_train, offensive_vocab = padded_everygram_pipeline(2, offensive['tweet'])
    lm_off = MLE(2)
    lm_off.fit(offensive_train, offensive_vocab)

    return lm_full, lm_not, lm_off


def test_LM(test_file_path, languageModel, results_file_name):
    test_data = pd.read_csv(test_file_path, sep='\t')
    test_data = preprocess(test_data)

    print(test_data.head())

    test_data_csv = pd.read_csv(test_file_path, sep='\t')
    test_data_csv['mle'] = test_data.iloc[:, 1].apply(lambda x: calculate_mle_for_tweet(languageModel, x))
    test_data_csv.to_csv("results/" + results_file_name + ".csv", sep='\t')
    print("Saved language model results to results/language_model_results.csv")


lm_full, lm_not, lm_off = train_LM("data/olid-training-v1.0.tsv")
test_LM("data/testset-levela.tsv", lm_full, "lm_full")
test_LM("data/testset-levela.tsv", lm_not, "lm_not")
test_LM("data/testset-levela.tsv", lm_off, "lm_off")