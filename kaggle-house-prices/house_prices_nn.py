import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# read in data
raw_data = pd.read_csv("data/train.csv")

# fill in missing values
# How many missing for each variable?
def count_missing(s):
    return np.sum(pd.isna(s))

missing_counts = raw_data.agg(count_missing)
missing_counts = missing_counts[missing_counts > 0]
missing_counts

raw_data.dtypes[missing_counts.index]

# fill lot frontage with median
raw_data['LotFrontage'] = raw_data['LotFrontage']. \
    fillna(raw_data['LotFrontage'].median())

# fill alley with "no_alley"
raw_data['Alley'] = raw_data['Alley']. \
    fillna('no_alley')

# fill MasVnrType with "None" (most common) and MasVnrArea with 0
raw_data['MasVnrType'] = raw_data['MasVnrType']. \
    fillna('None')
raw_data['MasVnrArea'] = raw_data['MasVnrType']. \
    fillna(0)

# fill FireplaceQu with 'no_fireplace' (Fireplaces=0 for all cases)
raw_data['FireplaceQu'] = raw_data['MasVnrType']. \
    fillna('no_fireplace')

# fill PoolQC with "no_poolqc"
raw_data['PoolQC'] = raw_data['PoolQC']. \
    fillna('no_poolqc')

# fill Fence with "no_fence"
raw_data['Fence'] = raw_data['Fence']. \
    fillna('no_fence')

# fill MiscFeature with "no_misc_feature"
raw_data['MiscFeature'] = raw_data['MiscFeature']. \
    fillna('no_misc_feature')

# fill things about basement and garage with most common values
other_missing = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
    'BsmtFinType2', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual',
    'GarageCond', 'Electrical']
for missing_col in other_missing:
    most_common = raw_data[missing_col].value_counts().index[0]
    raw_data[missing_col] = raw_data[missing_col].fillna(most_common)




# get X into numeric encoding; one-hot encoding of categorical variables
def to_one_hot(values):
    unique_values = np.unique(values)
    n_unique = len(unique_values)
    result = np.zeros((len(values), n_unique))
    for i, value in enumerate(values):
        result[i, unique_values == value] = 1
    return result

encoded_X_dim = 0
for i in range(1, raw_data.shape[1] - 1):
    if raw_data.dtypes[i] == 'int64' or raw_data.dtypes[i] == 'float64':
        encoded_X_dim += 1
    else:
        encoded_X_dim += len(np.unique(raw_data.iloc[:, i]))

encoded_X = np.empty((raw_data.shape[0], encoded_X_dim))
start_col = 0
numeric_cols = []
numeric_cols_orig = []
for i in range(1, raw_data.shape[1] - 1):
    if raw_data.dtypes[i] == 'int64' or raw_data.dtypes[i] == 'float64':
        encoded_X[:, start_col] = raw_data.iloc[:, i]
        numeric_cols.append(start_col)
        numeric_cols_orig.append(i)
        start_col += 1
    else:
        num_new_cols = len(np.unique(raw_data.iloc[:, i]))
        encoded_X[:, start_col:(start_col + num_new_cols)] = \
            to_one_hot(raw_data.iloc[:, i])
        start_col += num_new_cols



# pull out y
y = raw_data['SalePrice']



# train/test split
np.random.seed(938)
shuffled_inds = np.arange(len(y))
np.random.shuffle(shuffled_inds)
train_size = int(np.floor(len(y) * 0.75))
subtrain_size = int(np.floor(train_size * 0.75))

train_x = encoded_X[shuffled_inds[:subtrain_size], :]
val_x = encoded_X[shuffled_inds[subtrain_size:train_size], :]
full_train_x = encoded_X[shuffled_inds[:train_size], :]
test_x = encoded_X[shuffled_inds[train_size:], :]

train_y = y[shuffled_inds[:subtrain_size]]
val_y = y[shuffled_inds[subtrain_size:train_size]]
full_train_y = y[shuffled_inds[:train_size]]
test_y = y[shuffled_inds[train_size:]]



def standardize_x(train_x, test_x, numeric_cols):
    train_mean = train_x[:, numeric_cols].mean(axis=0)
    train_std = train_x[:, numeric_cols].std(axis=0)
    
    train_x = train_x.copy()
    test_x = test_x.copy()
    train_x[:, numeric_cols] -= train_mean
    train_x[:, numeric_cols] /= train_std
    
    test_x[:, numeric_cols] -= train_mean
    test_x[:, numeric_cols] /= train_std
    
    return((train_x, test_x))


# define model 1: 2 hidden layers of 4 units each.
(std_train_x, std_val_x) = standardize_x(train_x, val_x, numeric_cols)


model = models.Sequential()
model.add(layers.Dense(64, activation = 'relu',
    input_shape = (std_train_x.shape[1],)))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(1))

model.compile(optimizer='rmsprop',
    loss='mse',
    metrics=['mse', 'mae'])


n_epochs = 250
history = model.fit(std_train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (std_val_x, val_y))

plt.plot(range(1, n_epochs+1),
    np.sqrt(history.history['mean_squared_error']),
    'bo',
    label = "Training RMSE")

plt.plot(range(1, n_epochs+1),
    np.sqrt(history.history['val_mean_squared_error']),
    'o',
    label = "Validation RMSE")
plt.title('Training and validation RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()

np.min(np.sqrt(history.history['val_mean_squared_error']))



from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression

regr = RandomForestRegressor(max_depth=2,
    random_state = 0,
    n_estimators = 100)

regr.fit(std_train_x, train_y)

val_preds = regr.predict(std_val_x)

rf_val_rmse = np.sqrt(np.mean((val_y - val_preds)**2))

