import argparse
import pandas as pd
from transformer import GPT, TabGPTConfig
from table_dataset import TableDataset
from column_encoders import *
from column_generators import *
from torch import optim
from torch.utils.data import dataloader

encoder_dict = {'text': TextEncoder, 'categorical': CategoricalEncoder, 'continuous': ContinuousEmbedding}
generator_dict = {'text': TextGenerator, 'categorical': CategoricalGenerator, 'continuous': ContinuousGenerator}

def train_model(file, batch_size, column_names, column_types, num_epochs):
    dataframe = pd.read_csv(file)
    dataset = TableDataset(dataframe, column_names, column_types)
    data_loader = dataloader(dataset, batch_size=batch_size)

    column_encoders, column_generators = dataset.get_encoder_generator(TabGPTConfig.n_embd)

    model = GPT(TabGPTConfig, column_encoders, column_generators)
    optimizer = model.configure_optimizers(TabGPTConfig)


    for epoch in range(num_epochs):

        total_loss = 0
        for columns in data_loader:
            values = [col[0] for col in columns]
            data_id = [col[1] for col in columns]
            data_type = [col[2] for col in columns]

            loss = model(values)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f'epoch {epoch}, loss {total_loss}')
        torch.save(model.state_dict(), f'tab_gpt_epoch{epoch}')



def run(args):
    train_model(file=args.csv_file, batch_size=args.batch_size, column_names=args.column_names, column_types=args.column_types)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file')
    parser.add_argument('--batch-size')
    parser.add_argument('--column-names', type=str, nargs='+')
    parser.add_argument('--column-types')
    args = parser.parse_args()

    run(args)