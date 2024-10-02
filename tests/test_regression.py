import os
import sys

# Get the directory path of the current Python file
file_dir = os.path.dirname(__file__)

# Add the parent directory to the system path
sys.path.append(os.path.join(file_dir, 'C:\\Users\\PC\\water_test'))
import numpy as np
import pandas as pd
from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader
from models.regression.simple_regression import BasicRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
def convert_dataloader_to_dataframe(dataloader):
    """
    Convert the data from a dataloader into a pandas DataFrame.
    
    :param dataloader: DataLoader containing the features and targets.
    :return: DataFrame containing the features and target.
    """
    features, targets = [], []
    for data in dataloader:
        x, y = data
        features.append(x.numpy())  # Chuyển đổi từ tensor sang numpy array
        targets.append(y.numpy())

    features = np.vstack(features)  # Kết hợp các mảng lại
    targets = np.concatenate(targets)
    
    # Tạo DataFrame từ mảng đặc trưng
    feature_names = [f'feature_{i}' for i in range(features.shape[1])]  # Tên cột đặc trưng
    df = pd.DataFrame(features, columns=feature_names)
    df['target'] = targets
    return df



# Load and split data from Excel files
files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)

# Chuyển đổi training_data và testing_data thành DataFrame
train_df = convert_dataloader_to_dataframe(get_simple_dataloader(training_data, shuffle=True))
test_df = convert_dataloader_to_dataframe(get_simple_dataloader(testing_data, shuffle=False))

# Chia đặc trưng và giá trị mục tiêu
X_train = train_df.drop(columns=['target']).values
y_train = train_df['target'].values
X_test = test_df.drop(columns=['target']).values
y_test = test_df['target'].values

# Khởi tạo mô hình hồi quy tuyến tính
model = BasicRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán lỗi trung bình bình phương
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Hàm chuyển đổi từ DataLoader sang DataFrame

num_points_to_show = 5
for i in range(num_points_to_show):
    print(f'True Value: {y_test[i]}, Predicted Value: {y_pred[i]}')