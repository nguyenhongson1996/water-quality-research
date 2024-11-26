from torch import nn
from models.regression.simple_lstm import BasicLSTM
from utils.data_utils import load_and_split_data
from utils.lstm_dataset import get_lstm_dataloader

hidden_dim = 50
num_layers = 1
output_dim = 1
seq_length = 3

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)
train_dataloader = get_lstm_dataloader(training_data, seq_length, shuffle=True)
test_dataloader = get_lstm_dataloader(testing_data, seq_length, shuffle=False)

first_x, first_y = train_dataloader.dataset[0]
input_dim = first_x.shape[-1]

model = BasicLSTM(input_dim=input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim, seq_length = seq_length)

num_epochs = 200
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.MSELoss(), lr=lr,
          scheduler_params=lr_params, patience=20)
