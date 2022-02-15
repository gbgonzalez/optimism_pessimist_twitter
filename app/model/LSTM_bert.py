import torch
from utils.utils import get_device_torch
from transformers import BertModel, BertTokenizer

class LSTM_bert(torch.nn.Module):
    def __init__(self, num_class, dropout_rate, bert_config='bert-base-uncased'):

        super(LSTM_bert, self).__init__()

        self.num_class = num_class
        self.bert_config = bert_config
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)
        self.bert = BertModel.from_pretrained(self.bert_config)
        self.dropout_rate = 0.3
        self.lstm_input_size = self.bert.config.hidden_size
        self.lstm_hidden_size = int(768 / 2)
        self.lstm = torch.nn.LSTM(input_size=self.lstm_input_size,
                                  hidden_size=self.lstm_hidden_size,
                                  bidirectional=True)
        self.dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.fc = torch.nn.Linear(in_features=2 * self.lstm_hidden_size,
                                  out_features=self.num_class)

    def forward(self, sents_tensor, masks_tensor):
        device = get_device_torch()
        bert_output = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
        encoded_layers = bert_output[0].permute(1, 0, 2)
        enc_hiddens, (last_hidden, last_cell) = self.lstm(encoded_layers)
        output_hidden = torch.cat((last_hidden[0, :, :], last_hidden[1, :, :]), dim=1)
        output_hidden = self.dropout(output_hidden)
        output = self.fc(output_hidden)
        return output