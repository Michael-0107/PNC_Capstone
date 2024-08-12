import torch
import torch.nn as nn

from Hypers import Config

class LstmCovModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LstmCovModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1)
        self.batchnorm1d = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout2 = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # 将形状从 (batch_size, 4, 24) 变为 (batch_size, 24, 4)
        x = self.conv1d(x)  # 经过 Conv1d 后形状为 (batch_size, hidden_size, 4)
        x = self.batchnorm1d(x)  # 批归一化
        x = self.relu(x)
        x = x.permute(0, 2, 1)  # 转置回 (batch_size, 4, hidden_size)
        x, _ = self.lstm(x)  # 经过 LSTM 后形状为 (batch_size, 4, hidden_size)
        x = self.dropout1(x)
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的输出，形状为 (batch_size, hidden_size)，然后经过全连接层
        x = self.dropout2(x)
        return x
    
if __name__ == "__main__":
    model = LstmCovModel(24)
    random_tensor = torch.rand(16, 8, 24)

    out = model(random_tensor)

    print(out.shape)