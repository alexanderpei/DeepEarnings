import os
import numpy as np
import pandas as pd

# This function will load all of the text files that were webscraped using get_ZacksEarnings.R and converts them to a
# Pandas dataframe.
# Assumes that there exists a directory 'DirtyZacksEarnings'.

foldIn  = 'DirtyZacksEarnings'
foldOut = 'CleanZacksEarnings'
pathIn  = os.path.join(os.getcwd(), foldIn)
pathOut = os.path.join(os.getcwd(), foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

for file in os.listdir(pathIn):
    if file.endswith('.txt'):
        print('Currently cleaning: ' + file)

        pathFile = os.path.join(pathIn, file)
        fid = open(pathFile, 'r')
        Lines = fid.readlines()

        # Creating an individual data frame for each stock. Not sure if this is the best way. Dataframe will have the
        # indices as the quarters
        dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')
        df = pd.DataFrame(columns=('DateAnnounced', 'Estimate', 'Reported', 'Surprise', 'pctSurprise', 'AMC'), index=dates)

        for line in Lines[1:]:

            line.strip()
            split = line.strip().split('' '')

            # Need to build the index for the data frame.
            tempQDate = split[2]
            tempQDate = pd.to_datetime(tempQDate, format='%m/%Y').to_period('Q')

            # Clean up the estimate and reported data
            if split[3] == '--':
                Estimate = np.nan
            else:
                Estimate = split[3].replace('$', '')
                Estimate = float(Estimate.replace(',', ''))

            if split[4] == '--':
                Reported = np.nan
            else:
                Reported = split[4].replace('$', '')
                Reported = float(Reported.replace(',', ''))

            # Calculate the surprise and the percent surprise
            Surprise = Reported - Estimate

            if np.isnan(Estimate) or np.isnan(Reported) or Estimate == 0:
                pctSurprise = np.nan
            else:
                pctSurprise = Surprise/abs(Estimate)*100

            # Announced before market open or after market close
            if split[7][:-1] == 'After Close':
                amc = True
            else:
                amc = False

            df.at[tempQDate, 'DateAnnounced'] = pd.to_datetime(split[1], format='%m/%d/%Y')
            df.at[tempQDate, 'Estimate'] = Estimate
            df.at[tempQDate, 'Reported'] = Reported
            df.at[tempQDate, 'Surprise'] = Surprise
            df.at[tempQDate, 'pctSurprise'] = pctSurprise
            df.at[tempQDate, 'AMC'] = amc

        # Can't save a file if it is a DOS file name
        Dos = ['CON', 'AUX', 'PRN', 'LST', 'NUL']
        if file[:-4] not in Dos:
            df.to_pickle(os.path.join(pathOut, file[:-4] + '.pk'))
