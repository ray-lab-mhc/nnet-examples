import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from keras import models
from keras import layers

# read in data
all_data = pd.read_csv("data/ionosphere.data", header=None)
all_data[34] = (all_data[34] == "g").astype(float)

# train/test split
np.random.seed(938)
shuffled_inds = np.arange(len(all_data))
np.random.shuffle(shuffled_inds)
train_size = int(np.floor(len(all_data) * 0.75))
subtrain_size = int(np.floor(train_size * 0.75))

train_x = all_data.iloc[shuffled_inds[:subtrain_size], :34].values
val_x = all_data.iloc[shuffled_inds[subtrain_size:train_size], :34].values
full_train_x = all_data.iloc[shuffled_inds[:train_size], :34].values
test_x = all_data.iloc[shuffled_inds[train_size:], :34].values

train_y = all_data.iloc[shuffled_inds[:subtrain_size], 34].values
val_y = all_data.iloc[shuffled_inds[subtrain_size:train_size], 34].values
full_train_y = all_data.iloc[shuffled_inds[:train_size], 34].values
test_y = all_data.iloc[shuffled_inds[train_size:], 34].values

# define model 1: 2 hidden layers of 4 units each.
model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

plt.plot(range(1, n_epochs+1), history.history['loss'], 'bo', label = "Training loss")
plt.plot(range(1, n_epochs+1), history.history['val_loss'], 'o', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, n_epochs+1), history.history['acc'], 'bo', label = "Training accuracy")
plt.plot(range(1, n_epochs+1), history.history['val_acc'], 'o', label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

np.max(history.history['val_acc'])



# define model 1: 3 hidden layers of 4 units each.
model = models.Sequential()
model.add(layers.Dense(4, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

plt.plot(range(1, n_epochs+1), history.history['loss'], 'bo', label = "Training loss")
plt.plot(range(1, n_epochs+1), history.history['val_loss'], 'o', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, n_epochs+1), history.history['acc'], 'bo', label = "Training accuracy")
plt.plot(range(1, n_epochs+1), history.history['val_acc'], 'o', label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

np.max(history.history['val_acc'])







# define model 3: 1 hidden layers of 8 units, second of 4 units.
model = models.Sequential()
model.add(layers.Dense(8, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

plt.plot(range(1, n_epochs+1), history.history['loss'], 'bo', label = "Training loss")
plt.plot(range(1, n_epochs+1), history.history['val_loss'], 'o', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, n_epochs+1), history.history['acc'], 'bo', label = "Training accuracy")
plt.plot(range(1, n_epochs+1), history.history['val_acc'], 'o', label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

np.max(history.history['val_acc'])





# define model 4: 1 hidden layers of 12 units, second of 6 units.
model = models.Sequential()
model.add(layers.Dense(12, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(6, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

plt.plot(range(1, n_epochs+1), history.history['loss'], 'bo', label = "Training loss")
plt.plot(range(1, n_epochs+1), history.history['val_loss'], 'o', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(range(1, n_epochs+1), history.history['acc'], 'bo', label = "Training accuracy")
plt.plot(range(1, n_epochs+1), history.history['val_acc'], 'o', label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

np.max(history.history['val_acc'])




# define model 5: 1 hidden layers of 16 units, second of 8 units.
model = models.Sequential()
model.add(layers.Dense(16, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

plt.plot(range(1, n_epochs+1), history.history['loss'], 'bo', label = "Training loss")
plt.plot(range(1, n_epochs+1), history.history['val_loss'], 'o', label = "Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(range(1, n_epochs+1), history.history['acc'], 'bo', label = "Training accuracy")
plt.plot(range(1, n_epochs+1), history.history['val_acc'], 'o', label = "Validation accuracy")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

np.max(history.history['val_acc'])



# test set performance based on model 4
# refit model 4: 1 hidden layers of 12 units, second of 6 units.
model = models.Sequential()
model.add(layers.Dense(12, activation = 'relu', input_shape = (34,)))
model.add(layers.Dense(6, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy'])

n_epochs = 250
history = model.fit(train_x,
    train_y,
    epochs = n_epochs,
    batch_size = 32,
    validation_data = (val_x, val_y))

np.max(history.history['val_acc'])

train_epochs = int(np.floor(np.median(np.arange(1, n_epochs+1)[
    history.history['val_acc'] == np.max(history.history['val_acc'])])))
train_epochs

history2 = model.fit(full_train_x,
    full_train_y,
    epochs = train_epochs,
    batch_size=32)

model.evaluate(test_x, test_y)
# 90.9% accurate on test set

vals, counts = np.unique(train_y, return_counts = True)
counts[1] / np.sum(counts)
