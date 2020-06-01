import os
import sys
import numpy as np
import pandas as pd
from HelperFunctions import getFeatList, AlexComputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import normalize

np.set_printoptions(precision=3, threshold=sys.maxsize)

# This script will load in the Pandas data frames into a numpy matrix.
# Step 1. Will load in the earnings pandas data frame
# Step 2. Will load in the fundamentals quarterly data
#   - This step will require a file 'list_my_quarterly_features.txt'. This file contains the list of features that
#   - that you want to use. These features need normalization (for example dividing the feature by the market
#   - capitalization so that it is a fair comparison across companies)
#   = This step will also require a file 'list_my_quarterly_features_nonorm.txt'. These features do not need to be
#   - normalized. For example some ratios or percentage features.
# Step 3. Impute the missing data points using knnimpute

foldInEarn = 'CleanZacksEarnings'
foldInSent = 'CleanSentiment'
foldOut    = 'TrainDataSentiment'
if AlexComputer(): # Use the full data set
    pathInEarn = os.path.join(os.getcwd(), foldInEarn)
    pathInSent = os.path.join(os.getcwd(), foldInSent)
    pathOut = os.path.join(os.getcwd(), foldOut)
else:
    pathInEarn = os.path.join(os.getcwd(), 'Data', foldInEarn)
    pathInSent = os.path.join(os.getcwd(), 'Data', foldInSent)
    pathOut = os.path.join(os.getcwd(), 'Data', foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

featList = ['CharCnt', 'ComplexWordCnt', 'lmDictCnt', 'lmNegCnt',
                   'lmPosCnt', 'lmStrongCnt', 'lmModCnt', 'lmWeakCnt', 'lmUncerCnt',
                   'lmLitigCnt', 'harvNegCnt']

# Date indicies for the data frame
dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')

# Counting the number of features and pre allocating the numpy array
nFeat = len(featList)
nData = len(os.listdir(pathInEarn)) * len(dates) # Number of total data points
X = np.zeros((nData, nFeat))
y = np.zeros((nData))
# Looping through all of the earnings data frames
count = 0
for file in os.listdir(pathInEarn):
    if file.endswith('.pk'):

        print('Currently cleaning: ' + file)

        # Read in the data frames
        dfEarn = pd.read_pickle(os.path.join(pathInEarn, file))
        haveData = 1
        try:
            dfSent = pd.read_pickle(os.path.join(pathInSent, file))
        except:
            haveData = 0

        if haveData:
            feat = dfSent.loc[dates, featList].to_numpy().astype(np.float)
            featNormFact = dfSent.loc[dates, 'CharCnt'].to_numpy().astype(np.float) # We will normalize features by num char
            # Any good way to vectorize this???
            for idxRow in range(len(dates)):
                feat[idxRow, :] /= featNormFact[idxRow]

            # Fill in the matrix
            startIdx = count * len(dates)
            endIdx = startIdx + len(dates)
            X[startIdx:endIdx, 0:nFeat] = feat

            # Gather the labels. However, some of the earnings data will be nans. We want to remove these samples.
            # If either estimate or reported is a nan, set label as nan. Remove them later along with sample.
            Estimate = dfEarn['Estimate'].to_numpy().astype(np.float)
            Reported = dfEarn['Reported'].to_numpy().astype(np.float)
            idxBad = np.union1d(np.where(np.isnan(Estimate)), np.where(np.isnan(Reported))) # Union of both, need both to be valid
            yTemp = (Reported > Estimate).astype(np.float) # need float type to set nan value
            yTemp[idxBad] = np.nan # Set these values to nan and then take care of them later
            y[startIdx:endIdx] = yTemp

        count += 1

# Remove the bad indexes where the earnings estimate/reported were nan values
idxBad = np.where(np.isnan(y))
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

print(X.shape, y.shape)

# Delete any samples that are all zeros. These are rows that weren't filled in.
idxBad = np.where(np.sum(X == 0, axis=1) == nFeat)[0]
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

# Delete any samples that contain nans.
idxBad = np.where(np.sum(np.isnan(X), axis=1) > 0)
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

X = X[:,1:11]

print(X.shape, y.shape)

X = normalize(X, axis=0)

print(X.shape, y.shape)

# Save the output
np.save(os.path.join(pathOut, 'X.npy'), X)
np.save(os.path.join(pathOut, 'y.npy'), y)

print(X.shape, y.shape)
print(np.sum(y))