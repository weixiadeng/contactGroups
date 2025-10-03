import os.path
import torch
import torch.utils.data as data

import numpy as np
import pandas as pd
import time
from scipy.stats import spearmanr

import transformers, datasets
from transformers import EsmModel, AutoTokenizer

transformers.logging.set_verbosity_error()

from tqdm import tqdm
import random
import itertools

ESMs = [ "facebook/esm2_t6_8M_UR50D" ,
         "facebook/esm2_t12_35M_UR50D" ,
         "facebook/esm2_t30_150M_UR50D" ,
         "facebook/esm2_t33_650M_UR50D" ,
         "facebook/esm2_t36_3B_UR50D" ]

import torch
import torch.nn as nn
import torch.nn.functional as F

# multi-fc layer
# Input → {Linear → ReLU → Dropout} x n → Output
class predict_layer(nn.Module):
    def __init__(self, input_dim, dense_units, dropout_rate, num_label):
        super(predict_layer, self).__init__()
        self.normalizer = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, dense_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(dense_units, num_label)
        self.num_label = num_label

    def forward(self, x):
        x = self.normalizer(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.output(x)  # raw output (logits)
        return x

class predict_layer2(nn.Module):
    def __init__(self, input_dim, dense_units, dropout_rate, num_label):
        super(predict_layer2, self).__init__()
        self.normalizer = nn.BatchNorm1d(input_dim)
        self.fc1 = nn.Linear(input_dim, dense_units)
        self.dropout1 = nn.Dropout(dropout_rate)

        # self.normalizer2 = nn.BatchNorm1d(dense_units) 
        # previous self.normalizer is tuned for the input dimension only
        # adding another normalizier does not increase performance
        self.fc2 = nn.Linear(dense_units, dense_units)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.output = nn.Linear(dense_units, num_label)
        self.num_label = num_label

    def forward(self, x):
        x = self.normalizer(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # x = self.normalizer2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.output(x)  # raw output (logits)
        return x

# generalized multi-layer predictor
class mpredict_layer(nn.Module):
    def __init__(self, input_dim, dense_units, dropout_rate, num_label, num_hidden_layers=1):
        super(mpredict_layer, self).__init__()

        self.normalizer = nn.BatchNorm1d(input_dim)

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, dense_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = dense_units  # output of this layer is input to next

        self.hidden_layers = nn.Sequential(*layers)
        self.output = nn.Linear(dense_units, num_label)
        self.num_label = num_label

    def forward(self, x):
        x = self.normalizer(x)
        x = self.hidden_layers(x)
        x = self.output(x)
        return x

class TabularDataset(data.Dataset):
    def __init__(self, df, target_col):
        self.X = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, target_col].values, dtype=torch.float32)
        if len(self.y.shape) == 1:
            self.y = self.y.unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

###################################################
# w/o ft procedure
###################################################
# 
# spearman, preds = up.test_predictor(m)
# spearman: 0.8030012634744585 
def test_predictor(model):
    df = pd.read_pickle("./notebooks/embedding/example_data/GB1_embedded/test_ESM2_8M.pkl")
    target_col = -1
    model.eval()
    x_test = torch.tensor(df.iloc[:, :-2].values, dtype=torch.float32).to(next(model.parameters()).device)
    y_test = df.iloc[:, target_col].values
    # next(model.parameters()).device::Gets the first parameter tensor to determine the device of parameters  (one is enough)
	
    with torch.no_grad():
        preds = model(x_test).cpu().numpy().squeeze()
		
    spearman = spearmanr(y_test, preds).correlation
    return spearman, preds

# model, spearman_scores = up.train_predictor(seed = 42)    
def train_predictor(epochs=240, lr=1e-4, epsilon=1e-7, batch=8, dropout=0.2, dense=32, seed=99, num_labels=1, num_hidden_layers = 1):
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load and split your data (placeholder - replace `read()` with your own function)
    df_train=pd.read_pickle("./notebooks/embedding/example_data/GB1_embedded/train_ESM2_8M.pkl")
    df_valid=pd.read_pickle("./notebooks/embedding/example_data/GB1_embedded/valid_ESM2_8M.pkl")
    df_test=pd.read_pickle("./notebooks/embedding/example_data/GB1_embedded/test_ESM2_8M.pkl")

    # Create datasets and loaders
    train_dataset = TabularDataset(df_train, target_col=-1)
    valid_dataset = TabularDataset(df_valid, target_col=-1)

    train_loader = data.DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_loader = data.DataLoader(valid_dataset, batch_size=batch, shuffle=False)

    # Model initialization
    input_dim = df_train.shape[1] - 2  # Assuming last 2 columns are labels/metadata
    print('num_hidden_layer: %d' % num_hidden_layers)
    #model = predict_layer(input_dim=input_dim, dense_units=dense, dropout_rate=dropout, num_label=num_labels)
    #model = predict_layer2(input_dim=input_dim, dense_units=dense, dropout_rate=dropout, num_label=num_labels)
    model = mpredict_layer(input_dim=input_dim, dense_units=dense, dropout_rate=dropout, num_label=num_labels, num_hidden_layers = num_hidden_layers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=epsilon)
    loss_fn = nn.MSELoss() if num_labels == 1 else nn.CrossEntropyLoss()

    # Track spearman scores
    spearman_scores = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)

            if num_labels == 1:
                loss = loss_fn(preds, yb)
            else: # categorical 
                loss = loss_fn(preds, yb.long().squeeze())

            loss.backward()
            optimizer.step()

        # Validation Spearman calculation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(yb.numpy())

        all_preds = np.vstack(all_preds).squeeze()
        all_targets = np.vstack(all_targets).squeeze()
        
        # Compute Spearman correlation
        score = spearmanr(all_targets, all_preds).correlation
        spearman_scores.append(score)

        print(f"Epoch {epoch+1}/{epochs} - Spearman: {score:.4f}")

    return model, spearman_scores


# df_seq: {columns: sequence, ....}
def _emb_esm(df, emb_type='prot', checkpoint='facebook/esm2_t33_650M_UR50D'):
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = EsmModel.from_pretrained(checkpoint, torch_dtype=torch.float16)
    model = model.to("cuda")
    model = model.half()
    
    # embedding
    emb = []
    for i in tqdm(range(0,len(df))):
        inputs = tokenizer(df["sequence"].loc[i], return_tensors="pt", max_length = 10000, truncation=True, padding=False).to("cuda")
        with torch.no_grad():
            if emb_type == 'prot':
                # .cpu() required for converting to NumPy
                emb.append(np.array(torch.mean( model(**inputs).last_hidden_state.cpu(), dim = 1)))
            else:
                out = np.array( model(**inputs).last_hidden_state.cpu()) # out = [batch_size, seq_len, hidden_dim]
                out = np.squeeze(out) # remove singleton dimension. here is to remove batch_size dimention if batch_size = 1
                out = out[1:-1, :] # each embedding: [CLS, res_emb, SEP], out[1:-1,:] remove the first and last special token embeddings
                emb.append(out)
    return emb
    #return df_emb

def test_emb(infile = './training data/SecStr/test.pkl', emb_type='prot'):
    df_seq = pd.read_pickle(infile)
    df_seq["sequence"]=df_seq["sequence"].str.replace('|'.join(["O","B","U","Z","J"]),"X",regex=True)
    df = df_seq[['sequence']].head(2)
    print(len(df['sequence'][0]), df['sequence'][0])
    print(len(df['sequence'][1]), df['sequence'][1])

    ret = _emb_esm(df, emb_type=emb_type)
    print('\n'.join(['%s: %s' % (str(i), str(ret[i].shape)) for i in range(len(ret))]))
    return ret
    #df_emb = pd.DataFrame(np.concatenate(ret))
    # concatenating multiple sources can lead to non-sequential indices
    # reset_index set a new sequential index; drop =True: remove the old index
    #df_emb.reset_index(drop = True, inplace = True)


if __name__=='__main__':
    cp.dispatch(__name__)