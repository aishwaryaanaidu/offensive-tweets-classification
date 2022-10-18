import string

import pandas as pd
import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE


def clean_tweets(df):
    punctuations = string.punctuation

    df.loc[:, 'tweet'] = df.tweet.str.replace('@USER', '')  # Remove mentions (@USER)
    df.loc[:, 'tweet'] = df.tweet.str.replace('URL', '')  # Remove URLs
    df.loc[:, 'tweet'] = df.tweet.str.replace('&amp', 'and')  # Replace ampersand (&) with and
    df.loc[:, 'tweet'] = df.tweet.str.replace('&lt', '')  # Remove &lt
    df.loc[:, 'tweet'] = df.tweet.str.replace('&gt', '')  # Remove &gt
    df.loc[:, 'tweet'] = df.tweet.str.replace('\d+', '')  # Remove numbers
    df.loc[:, 'tweet'] = df.tweet.str.lower()  # Lowercase

    # Remove punctuations
    for punctuation in punctuations:
        df.loc[:, 'tweet'] = df.tweet.str.replace(punctuation, '')

    df.loc[:, 'tweet'] = df.astype(str).apply(
        lambda x: x.str.encode('ascii', 'ignore').str.decode('ascii')
    )
    # Remove emojis
    df.loc[:, 'tweet'] = df.tweet.str.strip()
    return df

def train_LM(train_file_path):
    df = pd.read_csv(train_file_path, sep='\t')
    # print(df['tweet'])
    # print((pd.Series(nltk.ngrams(df['tweet'], 2))))
    tweets = clean_tweets(df)
    # print(tweets.head())
    train, vocab = padded_everygram_pipeline(2, tweets['tweet'])
    print(train)
    lm = MLE(2)
    print(lm.vocab)
    lm.fit(train, vocab)
    print(lm.vocab)
    print(lm.vocab.lookup(["americans", "this"]))

train_LM("data/olid-training-v1.0.tsv")