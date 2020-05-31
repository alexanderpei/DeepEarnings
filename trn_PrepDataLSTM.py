import os
import sys
import numpy as np
import pandas as pd
from HelperFunctions import AlexComputer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import normalize
np.set_printoptions(precision=3, threshold=sys.maxsize)

# This script will load in the Pandas data frames into a numpy matrix.
# Step 1. Will load in the earnings pandas data frame
# Step 2. Will load in the stock data
#   - This step will require a file 'list_my_quarterly_features.txt'. This file contains the list of features that
#   - that you want to use. These features need normalization (for example dividing the feature by the market
#   - capitalization so that it is a fair comparison across companies)
#   = This step will also require a file 'list_my_quarterly_features_nonorm.txt'. These features do not need to be
#   - normalized. For example some ratios or percentage features.
# Step 3. Impute the missing data points using knnimpute

foldInEarn = 'CleanZacksEarnings'
foldInPrce = 'CleanStockPrice'
foldOut    = 'TrainDataLSTM'
if AlexComputer(): # Use the full data set
    pathInEarn = os.path.join(os.getcwd(), foldInEarn)
    pathInPrce = os.path.join(os.getcwd(), foldInPrce)
    pathOut = os.path.join(os.getcwd(), foldOut)
else:
    pathInEarn = os.path.join(os.getcwd(), 'Data', foldInEarn)
    pathInPrce = os.path.join(os.getcwd(), 'Data', foldInPrce)
    pathOut = os.path.join(os.getcwd(), 'Data', foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

# Stock data feature list. Feel free to customize your own feature list
featList = ['bidlo', 'askhi', 'prc', 'vol', 'bid', 'ask', 'openprc', 'retx']

# Date indicies for the earnings data frame
dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')

# For the time series, we are going to look back at the past 30 days of trading before the earnings announcement
nDay = 30

# Pre allocating the numpy array. Keras LSTM 3D tensor with shape [batch, timesteps, feature]
nFeat = len(featList)
nData = len(os.listdir(pathInEarn)) * len(dates) # Number of total data points
X = np.zeros((nData, nDay, nFeat))
y = np.zeros((nData))

# Looping through all of the earnings data frames
count = 0
for file in os.listdir(pathInEarn):
    if file.endswith('.pk'):

        print('Currently cleaning: ' + file)

        # Read in the data frames
        dfEarn = pd.read_pickle(os.path.join(pathInEarn, file))

        haveData = 1
        try: # Do we have stock data?
            dfPrce = pd.read_pickle(os.path.join(pathInPrce, file))
        except:
            haveData = 0

        for date in dates:
            announceDate = dfEarn.at[date, 'DateAnnounced']
            if not isinstance(announceDate, np.float): # Not a nan
                # Consider every trading day 30 days before the announcement. If announced BMO, need to shift that index
                # back by 1. However, we will overshoot the index to 90 days and trim later, to account for weekend
                # dates. Adjust parameters to your liking.
                if not dfEarn.at[date, 'AMC']:
                    announceDate -= pd.Timedelta(1, unit='D')

                endDate        = announceDate
                startDate      = announceDate - pd.Timedelta(90, unit='D') # 60 days before announcement
                stockTimeRange = pd.date_range(startDate, endDate, freq='D')

                # If there is no data, set y[count] = np.nan and the sample will get deleted later
                if haveData:

                    # Sometimes there will be a random character in the stock price that cannot be converted to a float.
                    # Not sure how to avoid this without try except
                    try:
                        tempData = dfPrce.loc[startDate:endDate, featList].to_numpy().astype(np.float)
                    except:
                        print('Random char in array')

                    # Delete rows which contain nans:
                    idxBad = np.where(np.sum(np.isnan(tempData), 1) > 0)
                    tempData = np.delete(tempData, idxBad, axis=0)
                    # If there are more than nDay data points, store the data. Also normalize across time points
                    # within each features.
                    if tempData.shape[0] >= nDay:
                        X[count, :, :] = normalize(tempData[-nDay:, :], axis=0)
                    else:
                        goodTempData = 0

                    Estimate = dfEarn.at[date, 'Estimate']
                    Reported = dfEarn.at[date, 'Reported']
                    if np.isnan(Estimate) or np.isnan(Reported):
                        y[count] = np.nan
                    elif Reported > Estimate:
                        y[count] = 1
                    else:
                        y[count] = 0
                else:
                    y[count] = np.nan

            count += 1

# Remove the bad indexes where the earnings estimate/reported were nan values. Also will remove samples where
# haveData = 0
print(X.shape, y.shape)
idxBad = np.where(np.isnan(y))
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

print(X.shape, y.shape)
# Remove samples where everything is a zero
idxBad = np.where(np.sum(X, axis=(1, 2)) == 0)
X = np.delete(X, idxBad, axis=0)
y = np.delete(y, idxBad)

# Save the output
np.save(os.path.join(pathOut, 'X.npy'), X)
np.save(os.path.join(pathOut, 'y.npy'), y)

print(X.shape, y.shape)
print(np.sum(y))

