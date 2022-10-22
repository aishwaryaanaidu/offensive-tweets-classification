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
    return text


def calculate_mle_for_tweet(language_model, data):
    mle_score = 0
    for word in data:
        mle_score += language_model.score(word)
    return round(mle_score, 2)


def preprocess(data):
    # Remove mentions (@USER)
    data.loc[:, 'tweet'] = data.tweet.str.replace('@USER', '')
    # Converting the text to lowercase
    data['tweet'] = data['tweet'].str.lower()

    # Remove stop words from each row
    stop_words = set(stopwords.words('english'))
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_stopwords(text, stop_words))

    # Removing all the punctuations
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_punctuations(text))

    # Cleaning numbers
    data['tweet'] = data['tweet'].apply(lambda text: cleaning_numbers(text))

    # Removing emojis
    data.loc[:, 'tweet'] = data.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )

    # Tokenization
    data['tweet'] = data['tweet'].apply(lambda text: tokenization(text))

    # Lemmatizing the text
    lemmatizer = nltk.WordNetLemmatizer()
    data['tweet'] = data['tweet'].apply(lambda text: lemmatizer_on_text(text, lemmatizer))

    return data


def train_LM(train_file_path):
    train_data = pd.read_csv(train_file_path, sep='\t')

    # Training on the whole dataset
    train_data = preprocess(train_data)
    train, vocab = padded_everygram_pipeline(2, train_data['tweet'])
    lm_full = MLE(2)
    lm_full.fit(train, vocab)

    # Training Non-offensive tweets
    non_offensive = train_data.loc[train_data['subtask_a'] == "NOT"]
    non_offensive_train, non_offensive_vocab = padded_everygram_pipeline(2, non_offensive['tweet'])
    lm_not = MLE(2)
    lm_not.fit(non_offensive_train, non_offensive_vocab)

    # Training Offensive tweets
    offensive = train_data.loc[train_data['subtask_a'] == "OFF"]
    offensive_train, offensive_vocab = padded_everygram_pipeline(2, offensive['tweet'])
    lm_off = MLE(2)
    lm_off.fit(offensive_train, offensive_vocab)

    return lm_full, lm_not, lm_off


def test_LM(test_file_path, language_model, results_file_name):
    test_data = pd.read_csv(test_file_path, sep='\t')
    # Pre-process the data
    test_data = preprocess(test_data)

    test_data_csv = pd.read_csv(test_file_path, sep='\t')
    print("Results of " + results_file_name + " model:")

    # Create a new column in the dataframe to capture MLE scores
    test_data_csv['mle'] = test_data.iloc[:, 1].apply(lambda x: calculate_mle_for_tweet(language_model, x))
    print("Average of MLE scores for " + results_file_name + " model", test_data_csv['mle'].mean())

    test_data_csv.to_csv("results/" + results_file_name + ".csv", sep='\t')
    print("Saved language model results to " + "results/" + results_file_name + ".csv\n")


# Train 3 models. One model will be trained on the whole dataset, the next model will be trained
# on the non-offensive tweets and the last one will be trained on the offensive tweets
lm_full, lm_not, lm_off = train_LM("data/olid-training-v1.0.tsv")

# test the testset with each of these models
test_LM("data/testset-levela.tsv", lm_full, "lm_full")
test_LM("data/testset-levela.tsv", lm_not, "lm_not")
test_LM("data/testset-levela.tsv", lm_off, "lm_off")