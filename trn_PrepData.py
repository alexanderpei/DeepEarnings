import os
import sys
import numpy as np
import pandas as pd
from HelperFunctions import getFeatList, AlexComputer
from sklearn.impute import KNNImputer
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
foldInFund = 'CleanCompustatFundQuarterly'
foldOut    = 'TrainData'
if AlexComputer(): # Use the full data set
    pathInEarn = os.path.join(os.getcwd(), foldInEarn)
    pathInFund = os.path.join(os.getcwd(), foldInFund)
    pathOut = os.path.join(os.getcwd(), foldOut)
else:
    pathInEarn = os.path.join(os.getcwd(), 'Data' foldInEarn)
    pathInFund = os.path.join(os.getcwd(), foldInFund)
    pathOut = os.path.join(os.getcwd(), foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

# Need to load in the feature lists
featList       = getFeatList('list_my_quarterly_features.txt')
featListNonorm = getFeatList('list_my_quarterly_features_nonorm.txt')

# Date indicies for the data frame
dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')

# Counting the number of features and pre allocating the numpy array
nFeat = len(featList) + len(featListNonorm)
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
        dfFund = pd.read_pickle(os.path.join(pathInFund, file))

        feat = dfFund[featList].to_numpy().astype(np.float)
        featNormFact = dfFund['mkvaltq'].to_numpy().astype(np.float) # We will normalize features by the market capitalization during each quarter
        # Any good way to vectorize this???
        for idxRow in range(len(dates)):
            feat[idxRow, :] /= featNormFact[idxRow]
        featNonorm = dfFund[featListNonorm].to_numpy().astype(np.float)

        # Fill in the matrix
        startIdx = count * len(dates)
        endIdx = startIdx + len(dates)
        X[startIdx:endIdx, 0:len(featList)] = feat
        X[startIdx:endIdx, len(featList):nFeat] = featNonorm

        count += 1

        # Gather the labels. However, some of the earnings data will be nans. We want to remove these samples.
        # If either estimate or reported is a nan, set label as nan. Remove them later along with sample.
        Estimate = dfEarn['Estimate'].to_numpy().astype(np.float)
        Reported = dfEarn['Reported'].to_numpy().astype(np.float)
        idxBad = np.union1d(np.where(np.isnan(Estimate)), np.where(np.isnan(Reported))) # Union of both, need both to be valid
        yTemp = (Reported > Estimate).astype(np.float) # need float type to set nan value
        yTemp[idxBad] = np.nan # Set these values to nan and then take care of them later
        y[startIdx:endIdx] = yTemp

# Remove the bad indexes where the earnings estimate/reported were nan values
idxBad = np.where(np.isnan(y))
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

print(np.sum(y))
print(X.shape, y.shape)

# Don't want samples that are too crap. Delete samples with more than 100 nans. Can adjust yourself.
# Also delete any samples that are all zeros. These are rows that weren't filled in.
idxBad = np.where(np.sum(np.isnan(X), axis=1) >= 100)[0]
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

print(X.shape, y.shape)

idxBad = np.where(np.sum(X == 0, axis=1) == nFeat)[0]
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

print(X.shape, y.shape)

# For the purpose of KNN imputation, need to have a feature where every sample has a number value (not a nan).
# Arbitrarily will choose the total long term debt as this feature. Then remove any row that doesn't have
# a value for this feature. Feel free to add more features.
idxNeedFeat = featList.index('dlttq')
idxBad = np.where(np.isnan(X[:, idxNeedFeat]))[0]
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad, axis=0)

print(X.shape, y.shape)

# Make features zero mean and unit variance
X = (X - np.nanmean(X, axis=0))/np.nanstd(X, axis=0)

# Remove any features which have nans across all of the samples
idxBad = np.where(np.sum(np.isnan(X), axis=0) == X.shape[0])[0]
X = np.delete(X, idxBad, axis=1)

print(X.shape, y.shape)

# Use KNN data imputation to fill in the nan values. This will take a while for large number of samples.
imputer = KNNImputer()
X = imputer.fit_transform(X)

# Save the output
np.savetxt(os.path.join(pathOut, 'X_imputed.txt'), X)
np.savetxt(os.path.join(pathOut, 'y_imputed.txt'), y)

print(X.shape, y.shape)
