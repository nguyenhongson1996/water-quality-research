from torch import nn
from models.regression.simple_regression import BasicCNN
from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)
train_dataloader = get_simple_dataloader(training_data, shuffle=True, batch_size = 4)
test_dataloader = get_simple_dataloader(testing_data, shuffle=False, batch_size= 4)

first_x, first_y = train_dataloader.dataset[0]

input_dim = first_x.shape[-1]

model = BasicCNN(input_dim = input_dim)

num_epochs = 500
lr = 0.01
lr_params = {"start_factor": 0.1,
             "end_factor": lr, "total_iters": num_epochs}
model.fit(train_dataloader, test_dataloader, epochs=num_epochs, optimizer_type="adam", loss_fn=nn.L1Loss(), lr=lr,
          scheduler_params=lr_params, patience=20)
