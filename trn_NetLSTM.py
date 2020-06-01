import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout, Dense, LSTM, Bidirectional
from keras.regularizers import l2
from HelperFunctions import AlexComputer, MakePlot
from sklearn.model_selection import train_test_split

foldIn  = 'TrainDataLSTM'
foldOut = 'NetLSTM'
if AlexComputer():
    pathIn  = os.path.join(os.getcwd(), foldIn)
    pathOut = os.path.join(os.getcwd(), foldOut)
else:
    pathIn  = os.path.join(os.getcwd(), 'Data', foldIn)
    pathOut = os.path.join(os.getcwd(), 'Data', foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

X = np.load(os.path.join(pathIn, 'X.npy'))
y = np.load(os.path.join(pathIn, 'y.npy'))
print(np.sum(y))

nTrain, nTimestep, nFeat = X.shape

print(X.shape)
print(y.shape)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)

model = Sequential()
model.add(Bidirectional(LSTM(10)))
model.add(Dropout(0.4))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, validation_split=0.1, epochs=100, batch_size=64, verbose=2)

# Plot
MakePlot(history)

_, trainAcc = model.evaluate(Xtrain, ytrain)
print('Train Accuracy: %.2f' % (trainAcc*100))
_, testAcc = model.evaluate(Xtest, ytest)
print('Test Accuracy: %.2f' % (testAcc*100))

# Save the model
pathOutModel = os.path.join(pathOut, 'model.json')
modelJson = model.to_json()
with open(pathOutModel, 'w') as jsonFile:
    jsonFile.write(modelJson)
model.save_weights(os.path.join(pathOut, 'model.h5'))