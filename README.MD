# Model to detect pessimism and optimism in tweets  

This repository provides the training and testing of two BERT models with Bi-LSTM. Each of the models uses a different corpus, i.e. the first corpus contains tweets manually labelled as optimistic and non-optimistic and the second corpus contains tweets manually labelled as pessimistic and non-pessimistic.

## Installation

The following section describes the installation process. This program has been tested on Python version 3.9. To download the repository as well as to install the necessary libraries, it is necessary to execute:

```bash
$ git clone https://github.com/gbgonzalez/optimism_pessimism_model.git
$ cd optimism_pessimism_model; pip install -r requirements.txt
```

## Execution

Once in the root folder of the repository it may be executed using the following command:
```bash
$ python app\run.py -batch_size 32 -lr 5e-5 -epochs 3 -max_len 128 -sent pessimism
```

<ul>
    <li> <b> batch_size: </b> argument that corresponds to the batch size with which the model will be trained </li>
    <li> <b> lr: </b> argument that corresponds to the learning rate with which the model will be trained</li>
    <li> <b> epochs: </b> number of epochs with which the model will be trained</li>
    <li> <b> max_len: </b> maximum length of each of the documents in the corpus</li>
    <li> <b> sent: </b> training corpus (pessimistic or optimistic)</li>
</ul>

All the information about the arguments used in the program can be displayed using:
```bash
$ python app\run.py -h
```

## Results

The best pessimism and optimism detection models achieved the following results:

Type Model | Accuracy | Recall | Precision | F-Score
------------- | ------------- | ------------- | ------------- | -------------
BERT-BiLSTM pessimism | 0.96 | 0.96 | 0.96 | 0.96
BERT-BiLSTM optimism | 0.92 | 0.94 | 0.91 | 0.92

