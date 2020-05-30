import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This function cleans the text file that was generated from querying  CRSP / Quarterly Update / Stock / Security Files
# / CRSP Daily Stock through Wharton Research Data Services. The data fields selected were all of them, but you really
# only need Company Name, Ticker, Price, Share Volume, Open Price, Ask or High, Bid or Low, Closing Bid, Closing Ask
# Return without dividends. Add more if you like
# The data is daily.
#
# Assumes that there exists a directory "DirtyStockPrice" with a text file in there from the query.

foldIn  = "DirtyStockPrice"
foldOut = "CleanStockPrice"
pathIn  = os.path.join(os.getcwd(), foldIn)
pathOut = os.path.join(os.getcwd(), foldOut)

for file in os.listdir(pathIn):
    if file.endswith(".txt"):
        print("Currently cleaning the CRSP Query")

        pathFile = os.path.join(pathIn, file)
        fid = open(pathFile, 'r')
        line = fid.readline()

        # These are all of the names of the queried variables that we will use for the data frame columns
        columns = line.split('\t')
        columns = [item.lower().strip() for item in columns]
        print(columns)

        # index for the data frame will be all of the days from 2010 to 2021
        dates = pd.date_range(start='1/1/2010', end='1/1/2021', freq='D')

        # The text file has each company stacked on top of each other
        prevTkr = '' # The ticker in the previous line, we need this to know when to make a new data frame
        while file:

            line = fid.readline().strip()
            split = line.split('\t')

            if len(split) > 2: # For some reason there are lines with no data
                # For our query, the ticker is the 3rd entry
                currTkr = split[2]
                if currTkr != prevTkr:
                    # Creating an individual data frame for each stock. Not sure if this is the best way. Dataframe will
                    # have the indices as the quarters. If we moved to a new ticker we will save the old one
                    print("Currently cleaning: " + currTkr)
                    if not prevTkr == '':
                        # Can't save a file if it is a DOS file name
                        Dos = ["CON", "AUX", "PRN", "LST", "NUL"]
                        if prevTkr not in Dos:

                            df.to_pickle(os.path.join(pathOut, prevTkr + ".pk"))
                    df = pd.DataFrame(columns=columns, index=dates)

                # For our query, the date is the 2nd entry
                tempDate = pd.to_datetime(split[1], format='%Y%m%d')

                # Fill in the data frame at this data. Sometimes there is missing data. For our query, the length of "split"
                # should be 12.
                if len(split) == 12:
                    df.at[tempDate, columns] = split

                prevTkr = currTkr

# Save the last data frame
df.to_pickle(os.path.join(pathOut, prevTkr + ".pk"))

