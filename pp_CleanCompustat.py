import os
import numpy as np
import pandas as pd

# This function cleans the text file that was generated from querying Compustat - Capital IQ / Compustat / North America
# - Daily / Fundamentals Quarterly through Wharton Research Data Services. The data fields selected were all of them.
# Assumes that there exists a directory 'DirtyCompustatFundQuarterly' with a text file in there from the query

foldIn  = 'DirtyCompustatFundQuarterly'
foldOut = 'CleanCompustatFundQuarterly'
pathIn  = os.path.join(os.getcwd(), foldIn)
pathOut = os.path.join(os.getcwd(), foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

for file in os.listdir(pathIn):
    if file.endswith('.txt'):
        print('Currently cleaning the Compustat Query')

        pathFile = os.path.join(pathIn, file)
        fid = open(pathFile, 'r')
        Lines = fid.readlines()

        # These are all of the names of the queried variables that we will use for the data frame columns
        dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')
        columns = Lines[0].split('\t')

        # The text file has each company stacked on top of each other
        prevTkr = '' # The ticker in the previous line, we need this to know when to make a new data frame
        for line in Lines[1:]:

            split = line.split('\t')
            # In this case, the ninth entry in split is the ticker queried variable from Compustat. May differ
            # depending on how you queried the database.
            currTkr = split[9]
            if currTkr != prevTkr:
                # Creating an individual data frame for each stock. Not sure if this is the best way. Dataframe will
                # have the indices as the quarters. If we moved to a new ticker we will save the old one
                print('Currently cleaning: ' + currTkr)
                if not prevTkr == '':
                    # Can't save a file if it is a DOS file name
                    Dos = ['CON', 'AUX', 'PRN', 'LST', 'NUL']
                    if prevTkr not in Dos:
                        df.to_pickle(os.path.join(pathOut, prevTkr + '.pk'))
                df = pd.DataFrame(columns=columns, index=dates)

            # In this case, the second entry in split is the datadate queried variable from Compustat. May differ
            # depending on how you queried the database.
            tempQDate = pd.to_datetime(split[1], format='%Y%m%d').to_period('Q')

            for i in range(len(split)):
                if split[i] == '':
                    split[i] = np.nan
            df.at[tempQDate, columns] = split

            prevTkr = currTkr

# Save the last data frame
df.to_pickle(os.path.join(pathOut, prevTkr + '.pk'))
