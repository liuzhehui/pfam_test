import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, num_classes, embedding_dim, dropout_rate=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)        
        self.bilstm = nn.LSTM(self.embedding_dim, self.num_classes, batch_first = True, bidirectional = True)
        self.linear1 = nn.Linear(4*self.num_classes, self.num_classes)
        self.linear2 = nn.Linear(self.num_classes, self.num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        
    def forward(self, batch_inputs):
        x = self.embedding(batch_inputs.type(torch.int64))
        x, (h, c) = self.bilstm(x)
        avg_pool = torch.mean(x, 1)
        max_pool, _ = torch.max(x, 1)
        conc = torch.cat((avg_pool, max_pool), 1)
        x = self.dropout(self.relu(self.linear1(conc)))
        x = self.linear2(x)
        x = self.softmax(x)
        return x