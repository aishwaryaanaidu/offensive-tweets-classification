import nltk
import pandas as pd
from nltk.corpus import stopwords
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

df = pd.DataFrame()
vectoriser = TfidfVectorizer()


def cleaning_stopwords(text, stop_words):
    return " ".join([word for word in str(text).split() if word not in stop_words])


def cleaning_punctuations(text):
    punctuations_list = string.punctuation
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)


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
        lambda text: text.str.encode('ascii', 'ignore').str.decode('ascii')
    )

    return data


def train_LR_model(train_file_path):
    data = pd.read_csv(train_file_path, sep='\t')
    data = preprocess(data)

    # Dropping unnecessary columns
    data.drop(['subtask_b', 'subtask_c'], axis=1, inplace=True)
    X_train = data['tweet']

    # Vectorizing dataset
    vectoriser.fit(X_train)
    X_train = vectoriser.transform(X_train)

    logistic_regression_model = LogisticRegression(C=10, max_iter=150)
    logistic_regression_model.fit(X_train, data['subtask_a'])

    return logistic_regression_model


def test_LR_model(test_file_path, LR_model):
    test_data = pd.read_csv(test_file_path, sep='\t')
    test_data = preprocess(test_data)

    X_test = vectoriser.transform(test_data['tweet'])
    predictions = LR_model.predict(X_test)
    probabilities = LR_model.predict_proba(X_test)

    test_data_csv = pd.read_csv(test_file_path, sep='\t')

    # Creating new columns in the dataframe to store probability and predictions
    test_data_csv["offensive_probability"] = probabilities[:, 1]
    test_data_csv["predictions"] = predictions
    test_data_csv.to_csv("results/logistic_regression_results.csv", sep='\t')

    print("Saved logistic regression model's results to results/logistic_regression_results.csv")

    actual_labels = pd.read_csv("data/labels-levela.csv", header=None)
    actual_labels = actual_labels.iloc[:, 1]

    score = accuracy_score(actual_labels, predictions)
    print("Accuracy: {}".format(score))


logistic_regression_model = train_LR_model("data/olid-training-v1.0.tsv")
test_LR_model("data/testset-levela.tsv", logistic_regression_model)