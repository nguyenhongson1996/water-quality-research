from torch import nn
from models.regression.simple_regression import BasicRegression
from models.regression.simple_lstm import BasicLSTM
from models.regression.simple_scikit import SimpleSkLearnRegression
from models.regression.simple_scikit import SimpleSVR
from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)
train_dataloader = get_simple_dataloader(training_data, shuffle=True)
test_dataloader = get_simple_dataloader(testing_data, shuffle=False)

first_x, first_y = train_dataloader.dataset[0]

input_dim = first_x.shape[-1]
hidden_dim = 50
num_layers = 1
output_dim = 1
# model = SimpleSkLearnRegression()
# model = SimpleSVR()
# model = BasicRegression(input_dim)
model = BasicLSTM(input_dim=input_dim, hidden_dim = hidden_dim, num_layers = num_layers, output_dim = output_dim)

num_epochs = 200
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.MSELoss(), lr=lr,
          scheduler_params=lr_params, patience=20)
