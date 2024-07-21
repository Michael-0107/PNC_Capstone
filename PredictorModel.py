import torch
import torch.nn as nn

from Hypers import Config


class PredictorModel(nn.Module):
    def __init__(self, input_size, hidden_size, proj_size=0):
        super(PredictorModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, proj_size=proj_size, dropout=0.2)

    def forward(self, x, h, c):
        out, (h_new, c_new) = self.lstm(x, (h, c))

        return out, h_new, c_new
    
    

if __name__ == "__main__":
    import Hypers
    model = PredictorModel(len(Hypers.feature_list), hidden_size=128, proj_size=len(Hypers.rating_to_category))
    random_tensor = torch.rand(4, 31, len(Hypers.feature_list))

    h = torch.zeros(1, 4, len(Hypers.feature_list))
    c = torch.zeros(1, 4, 128)

    out, h_new, c_new = model(random_tensor, h, c)

    print(out.shape)
    print(h_new.shape)
    print(c_new.shape)
    


