import torch
import torch.nn as nn

from Hypers import Config


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, proj_size=0):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, proj_size=proj_size, dropout=0.2)
        
        self.linear = nn.Linear(hidden_size, 1)
        if proj_size != 0:
            self.linear = nn.Linear(proj_size, 1)


    def forward(self, x):
        out, (h_new, c_new) = self.lstm(x)
        out = self.linear(out)

        return out
    
    

if __name__ == "__main__":
    import Hypers
    model = LSTMModel(len(Hypers.feature_list), hidden_size=128, proj_size=len(Hypers.rating_to_category))
    random_tensor = torch.rand(4, 31, len(Hypers.feature_list))

    h = torch.zeros(1, 4, len(Hypers.feature_list))
    c = torch.zeros(1, 4, 128)

    out, h_new, c_new = model(random_tensor, h, c)

    print(out.shape)
    print(h_new.shape)
    print(c_new.shape)
    


