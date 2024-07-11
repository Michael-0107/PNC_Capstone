import torch
import torch.nn as nn



class PredictorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictorModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
    


if __name__ == "__main__":
    pass

