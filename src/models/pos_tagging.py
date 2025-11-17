import torch
import torch.nn as nn


class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.out = nn.Linear(hidden_size, output_dim)
        
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.emb(x)                      # (batch, seq_len, embed)
        output, (h_n, c_n) = self.rnn(x)     # output: (batch, seq_len, hidden)
        logits = self.out(output)            # (batch, seq_len, output_dim)
        
        return logits
