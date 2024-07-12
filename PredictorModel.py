import torch
import torch.nn as nn

import Hypers

class PredictorModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PredictorModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x, h, c):
        out, (h_new, c_new) = self.lstm(x, (h, c))

        return out, h_new, c_new
    
    


if __name__ == "__main__":
    model = PredictorModel(len(Hypers.feature_list), 20, len(Hypers.rating_to_category))
    random_tensor = torch.rand(4, 31, len(Hypers.feature_list))

    h = torch.zeros(1, 4, 20)
    c = torch.zeros(1, 4, 20)

    out, h_new, c_new = model(random_tensor, h, c)

    print(out.shape)
    print(h_new.shape)
    print(c_new.shape)
    


