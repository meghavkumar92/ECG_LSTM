import torch
from torch import nn, optim
import torch.nn.functional as F
import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np

SEQ_LEN = 140
N_FEATURES = 1
EMBEDDING_DIM = 128
MODEL_PATH = "E:\\Programs\\Megha\\ARP_A\\Physionet\\PTB_ECG\\"
MODEL_DICT = {'M1': 'model_qrs_lstm2.pth', 'M2': 'model_qrs_lstm.pth'}

class Encoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64):
        super(Encoder, self).__init__()

        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

        self.rnn1 = nn.LSTM(
          input_size=n_features,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=self.hidden_dim,
          hidden_size=embedding_dim,
          num_layers=1,
          batch_first=True
        )
        
        self.linear_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, x):
        #print(x.shape)
        x = x.reshape((-1, self.seq_len, self.n_features))

        x, (_, _) = self.rnn1(x)
        #print(x.shape,h1_n.shape);#
        #x = nn.sigmoid(self.linear_layer(x))
        x, (hidden_n, _) = self.rnn2(x)
        #print(x.shape, hidden_n.shape);#, hidden_n.reshape((-1,self.n_features, self.embedding_dim)).shape)
        
        return hidden_n.reshape((-1,self.n_features, self.embedding_dim))

class Decoder(nn.Module):

    def __init__(self, seq_len, input_dim=64, n_features=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_dim = seq_len, input_dim
        self.hidden_dim, self.n_features = 2 * input_dim, n_features

        self.rnn1 = nn.LSTM(
          input_size=input_dim,
          hidden_size=input_dim,
          num_layers=1,
          batch_first=True
        )

        self.rnn2 = nn.LSTM(
          input_size=input_dim,
          hidden_size=self.hidden_dim,
          num_layers=1,
          batch_first=True
        )

        #self.linear_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.output_layer = nn.Linear(self.hidden_dim, n_features)

    def forward(self, x):
        #bsz,_,_ = x.shape
        #print(x.shape);
        x = x.repeat(1,self.seq_len, self.n_features)
        #print(x.shape);
        x = x.reshape((-1, self.seq_len, self.input_dim))

        x, (hidden_n, cell_n) = self.rnn1(x)
        x, (hidden_n, cell_n) = self.rnn2(x)
        x = x.reshape((-1,self.seq_len, self.hidden_dim))

        return self.output_layer(x)

class RecurrentAutoencoder(nn.Module):

    def __init__(self, seq_len, n_features, embedding_dim=64, device='cpu'):
        super(RecurrentAutoencoder, self).__init__()

        self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
        self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def load_model(model_name='M1',device='cpu'):
    model = RecurrentAutoencoder(SEQ_LEN, N_FEATURES, EMBEDDING_DIM, device)
    print("Loading model...")
    model.load_state_dict(torch.load(MODEL_PATH+MODEL_DICT[model_name], map_location=device))
    model.eval()
    print("Model loaded!!")
    return model

def get_dataloader(df):

    sequences = df.iloc[:,::5].astype(np.float32).to_numpy().tolist()

    dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]

    n_seq, seq_len, n_features = torch.stack(dataset).shape
    
    print(n_seq, seq_len, n_features)

    test_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=5)

    return test_dataloader, seq_len, n_features

def get_predictions(model, dataloader, device='cpu'):
    predictions, losses = [], []
    criterion = nn.SmoothL1Loss(reduction='sum').to(device)
    with torch.no_grad():
        model = model.eval()
        for seq_true in dataloader:
            #print(seq_true.shape);
            seq_true = seq_true.to(device)
            seq_pred = model(seq_true)

            loss = criterion(seq_pred, seq_true)

            predictions.append(seq_pred.cpu().numpy().flatten())
            losses.append(loss.item())
    return np.array(predictions), np.array(losses)