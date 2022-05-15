from responses import target
import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class CategoricalGenerator:
    def __init__(self, num_categories, embedding_dim):
        self.prediction_layer = nn.Linear(embedding_dim, num_categories)

    def forward(self, representation, labels=None):
        return self.prediction_layer(representation)



class TextGenerator:
    def __init__(self, embedding_dim, model_name, gpt_dim=768):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)

        self.dim_adapter = nn.Sequential(
            nn.Linear(embedding_dim, gpt_dim),
            nn.ReLU(),
            self.Linear(gpt_dim, gpt_dim)
        )

    def forward(self, representation, target_texts=None):
        if target_texts:
            target_tokenized = self.gpt2_tokenizer(target_texts, max_length=512, truncation=True, padding=True, return_tensors='pt')
        else:
            target_tokenized = None

        loss = self.gpt(inputs_embeds=representation, labels=target_tokenized).loss

        return loss, loss



class ContinuousGenerator:
    def __init__(self, embedding_dim):
        self.prediction_layer = nn.Linear(embedding_dim, 1)

    def forward(self, representation):
        return self.prediction_layer(representation)


