import argparse
import torch
import os
from processing import load_data
from model import dataloader_bert, train_bilstm, test_model
from utils.utils import set_seed

if __name__ == "__main__":

    #get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", "--batch_size", type=str, help="Batch size")
    parser.add_argument("-lr", "--learning_rate", type=str, help="Learning rate in training model")
    parser.add_argument("-epochs", "--epochs", type=str, help="Number of epcohs for train")
    parser.add_argument("-max_len", "--max_len", type=str, help="Maximum length of each of the documents in the corpus")
    parser.add_argument("-sent", "--sentiment", type=str, help="Choose type of corpus")
    args = parser.parse_args()

    # Model Parameters
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    epochs = int(args.epochs)
    MAX_LEN = int(args.max_len)
    sent = args.sentiment

    #Global variables
    model_route = f"data/{sent}/model"
    csv_train_route = f"data/{sent}/train.csv"
    csv_test_route = f"data/{sent}/train.csv"
    train_pickle_route = f"data/{sent}/train.pkl"
    test_pickle_route = f"data/{sent}/test.pkl"
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    #Credential Twitter API
    credentials = {
        "CONSUMER_KEY": "XXXX",
        "CONSUMER_SECRET": "XXXX",
        "ACCESS_TOKEN": "XXXX",
        "ACCESS_SECRET": "XXXX"
    }

    ## Rehydrated tweets
    dataset_train, dataset_test = load_data(credentials, csv_train_route, csv_test_route, train_pickle_route,
                                            test_pickle_route, sent)

    set_seed(2021)

    for id in dataset_train.keys():
        X_train.append(dataset_train[id]["text"])
        y_train.append(dataset_train[id]["sent"])

    if not os.path.isfile(model_route):
        print("Start training.....")
        train_data, train_sampler, train_dataloader = dataloader_bert(MAX_LEN, batch_size, X_train, y_train)
        train_bilstm(epochs, learning_rate, train_dataloader, model_route)

    print("Start test.....")
    model = torch.load(model_route)
    for id in dataset_test.keys():
        X_test.append(dataset_test[id]["text"])
        y_test.append(dataset_test[id]["sent"])

    test_data, test_sampler, test_dataloader = dataloader_bert(MAX_LEN, batch_size, X_test, y_test)
    auc, threshold, accuracy, precision, recall, f1 = test_model(model, test_dataloader)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
