import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.regularizers import l2
from HelperFunctions import AlexComputer, MakePlot
from sklearn.model_selection import train_test_split

foldIn  = 'TrainData'
foldOut = 'Net'
if AlexComputer():
    pathIn  = os.path.join(os.getcwd(), foldIn)
    pathOut = os.path.join(os.getcwd(), foldOut)
else:
    pathIn  = os.path.join(os.getcwd(), 'Data', foldIn)
    pathOut = os.path.join(os.getcwd(), 'Data', foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

X = np.loadtxt(os.path.join(pathIn, 'X_imputed.txt'))
y = np.loadtxt(os.path.join(pathIn, 'y_imputed.txt'))

nTrain, inDim = X.shape

print(X.shape)
print(y.shape)

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, shuffle=False)

# model = Sequential()
# model.add(Dense(250, input_dim=inDim, activation='selu'))
# model.add(Dropout(0.2))
# model.add(Dense(250, activation='selu', kernel_regularizer=l2(0.1), bias_regularizer=l2(0.1)))
# model.add(Dense(1, activation='sigmoid'))

model = Sequential()
model.add(Dense(100, input_dim=inDim, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.3), bias_regularizer=l2(0.3)))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
history = model.fit(Xtrain, ytrain, validation_split=0.1, epochs=500, batch_size=64, verbose=2)

# Do plot
MakePlot(history)

_, trainAcc = model.evaluate(Xtrain, ytrain)
print('Train Accuracy: %.2f' % (trainAcc*100))
_, testAcc = model.evaluate(Xtest, ytest)
print('Test Accuracy: %.2f' % (testAcc*100))

# Save the model
# pathOutModel = os.path.join(pathOut, 'model.json')
# modelJson = model.to_json()
# with open(pathOutModel, 'w') as jsonFile:
#     jsonFile.write(modelJson)
# model.save_weights(os.path.join(pathOut, 'model.h5'))