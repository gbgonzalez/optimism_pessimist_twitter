import pandas as pd
import tweepy
import pickle
import os

def load_data(credentials, csv_train_route, csv_test_route, train_pickle_route, test_pickle_route, sent):

    csv_train = pd.read_csv(csv_train_route, sep="\t")
    csv_test = pd.read_csv(csv_test_route, sep="\t")

    auth = tweepy.OAuthHandler(credentials["CONSUMER_KEY"], credentials["CONSUMER_SECRET"])
    auth.set_access_token(credentials["ACCESS_TOKEN"], credentials["ACCESS_SECRET"])
    api = tweepy.API(auth, wait_on_rate_limit=True)


    if not os.path.isfile(train_pickle_route):
        dataset_train = {}
        for i, id in enumerate(csv_train["ID"]):
            tweet = api.get_status(id, tweet_mode="extended")
            tweet_sent = 0
            if csv_train["SENTIMENT"][i] == sent:
                tweet_sent = 1
            dataset_train[id] = {"text": tweet.full_text, "sent": tweet_sent}

        pickle.dump(dataset_train, open(train_pickle_route, 'wb'))

    if not os.path.isfile(test_pickle_route):
        dataset_test = {}
        for i, id in enumerate(csv_test["ID"]):
            tweet = api.get_status(id, tweet_mode="extended")
            tweet_sent = 0
            if csv_test["SENTIMENT"][i] == sent:
                tweet_sent = 1
            dataset_test[id] = {"text": tweet.full_text, "sent": tweet_sent}

        pickle.dump(dataset_test, open(test_pickle_route, 'wb'))

    dataset_train = pickle.load(open(train_pickle_route, 'rb'))
    dataset_test = pickle.load(open(test_pickle_route, 'rb'))

    return dataset_train, dataset_test