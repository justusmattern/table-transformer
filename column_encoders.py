import torch
from torch import nn
from transformers import BertTokenizer, BertModel

class CategoricalEncoder:
    def __init__(self, num_categories, embedding_dim):
        self.embedding = nn.Embedding(num_categories, embedding_dim)

    def forward(self, category_index):
        return self.embedding(category_index)



class TextEncoder:
    def __init__(self, embedding_dim, trainable, bertmodel):
        self.bert_tokenizer = BertTokenizer.from_pretrained(bertmodel)
        self.bert = BertModel.from_pretrained(bertmodel)
        self.dim_adapter = nn.Linear(1, embedding_dim)

        if not trainable:
            for param in self.bert.parameters():
                param.requires_grad = False


    def forward(self, texts):
        tokenized_text = self.bert_tokenizer(texts, max_length=512, truncation=True, padding=True, return_tensors='pt')
        encoding = self.bert(tokenized_text).pooler_output
        out = self.dim_adapter(encoding)

        return out



class ContinuousEmbedding:
    def __init__(self, embedding_dim):
        self.expansion = nn.Linear(1, embedding_dim)

    def forward(self, value):
        return self.expansion(value)


