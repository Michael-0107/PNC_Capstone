import torch
import torch.nn as nn

from Hypers import Config


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, proj_size=0,num_classes=24):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, proj_size=proj_size, dropout=0.2)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        if proj_size != 0:
            self.fc = nn.Linear(proj_size, num_classes)


    def forward(self, x):
        out, (h_new, c_new) = self.lstm(x)
        # mask = mask.unsqueeze(-1).expand_as(out)
        # out = out * mask
        out = self.fc(out)  
        return out
    
    

if __name__ == "__main__":
    import Hypers
    model = LSTMModel(len(Hypers.feature_list), hidden_size=128, proj_size=len(Hypers.rating_to_category))
    random_tensor = torch.rand(32, 28, 96)

    h = torch.zeros(1, 32, 128)
    c = torch.zeros(1, 32, 128)

    out, h_new, c_new = model(random_tensor, h, c)

    print(out.shape)
    print(h_new.shape)
    print(c_new.shape)
    


