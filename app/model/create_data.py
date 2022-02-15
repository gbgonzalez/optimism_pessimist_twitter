from transformers import BertModel, BertTokenizer
import re
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



def dataloader_bert(MAX_LEN, batch_size, X_train, y_train):
    train_inputs, train_masks = preprocessing_for_bert(X_train, MAX_LEN)
    train_labels = torch.tensor(y_train)
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_data, train_sampler, train_dataloader

def tweet_preprocesing_bert(text):
    text = re.sub(r'(@.*?)[\s]', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocessing_for_bert(data, MAX_LEN):
    input_ids = []
    attention_masks = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=tweet_preprocesing_bert(sent),
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation = True
            )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

