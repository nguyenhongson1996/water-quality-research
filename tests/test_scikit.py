import numpy as np
import pandas as pd
from utils.data_utils import load_and_split_data
from utils.dataset import get_simple_dataloader
from models.regression.simple_scikit import SimpleSkLearnRegression
from sklearn.metrics import mean_squared_error
def convert_dataloader_to_dataframe(dataloader) -> pd.DataFrame:
    """
    Convert the data from a dataloader into a pandas DataFrame.
    
    :param dataloader: DataLoader containing the features and targets.
    :return: DataFrame containing the features and target.
    """
    features, targets = [], []
    for data in dataloader:
        x, y = data
        features.append(x.numpy())  
        targets.append(y.numpy())

    features = np.vstack(features) 
    targets = np.concatenate(targets)
    feature_names = [f'feature_{i}' for i in range(features.shape[1])]  
    df = pd.DataFrame(features, columns=feature_names)
    df['target'] = targets
    return df

files = ["predata.xls", "Data.xlsx"]
training_data, testing_data = load_and_split_data(files)
train_df = convert_dataloader_to_dataframe(get_simple_dataloader(training_data, shuffle=True))
test_df = convert_dataloader_to_dataframe(get_simple_dataloader(testing_data, shuffle=False))

X_train = train_df.drop(columns=['target']).values
y_train = train_df['target'].values
X_test = test_df.drop(columns=['target']).values
y_test = test_df['target'].values

model = SimpleSkLearnRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

num_points_to_show = 5
for i in range(num_points_to_show):
    print(f'True Value: {y_test[i]}, Predicted Value: {y_pred[i]}')