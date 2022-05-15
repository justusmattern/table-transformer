import torch
import pandas
from pandas import DataFrame
import random
from column_encoders import *
from column_generators import *

class TableDataset(torch.utils.data.Dataset):
    def __init__(self, data: DataFrame, column_names: list, column_types: list):
        super().__init__()

        column_ids = dict()
        for i, col in enumerate(column_names):
            column_ids[col] = i

        self.column_ids = column_ids
        self.column_names = column_names
        self.column_types = column_types

        self.data = data
        self.columns = []
        for col_name, col_type in zip(column_names, column_types):
            self.columns.append((data[col_name].to_list(), column_ids[col_name], col_type))

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        items = [(column[index], data_id, data_type) for column, data_id, data_type in self.columns]

        return items

    def get_encoder_generator(self, embedding_dim):
        encoders = []
        generators = []
        for col_name, col_type in zip(self.column_names, self.column_types):

            if col_type == 'text':
                enc = TextEncoder(embedding_dim=embedding_dim, trainable=True, bertmodel='bert-base-uncased')
                gen = TextGenerator(embedding_dim=embedding_dim, model_name='gpt2-large')
                encoders.append(enc)
                generators.append(gen)
            
            elif col_type == 'categorical':
                num_categories = len(set(self.data[col_name]))
                enc = CategoricalEncoder(num_categories=num_categories, embedding_dim=embedding_dim)
                gen = CategoricalGenerator(num_categories=num_categories, embedding_dim=embedding_dim)
                encoders.append(enc)
                generators.append(gen)
            
            elif col_type == 'continuous':
                enc = ContinuousEmbedding(embedding_dim=embedding_dim)
                gen = ContinuousGenerator(embedding_dim=embedding_dim)
                encoders.append(enc)
                generators.append(gen)
            
            else:
                raise Exception(f"{col_type} is not a valid column type.")
        
        return encoders, generators


        
