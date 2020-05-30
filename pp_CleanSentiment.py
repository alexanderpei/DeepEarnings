import os
import pandas as pd

# This function will load all of the text files that were gathered get_Sentiment.R and converts them to a
# Pandas data frame.
# Assumes that there exists a directory "DirtySentiment".

foldIn  = "DirtySentiment"
foldOut = "CleanSentiment"
pathIn  = os.path.join(os.getcwd(), foldIn)
pathOut = os.path.join(os.getcwd(), foldOut)

# Make the output directory if it doesn't exist
if not os.path.isdir(pathOut):
    os.mkdir(pathOut)

for file in os.listdir(pathIn):
    if file.endswith(".txt"):
        print("Currently cleaning: " + file)

        pathFile = os.path.join(pathIn, file)
        fid = open(pathFile, 'r')
        Lines = fid.readlines()

        # Creating an individual data frame for each stock. Not sure if this is the best way. Dataframe will have the
        # indices as the quarters
        dates = pd.date_range(start='1/1/2008', end='1/1/2021', freq='Q').to_period('Q')
        columns = ['CIK', 'CompanyName', 'FormType', 'DateFiled', 'AccessionNumber',
                   'FileSize', 'CharCnt', 'ComplexWordCnt', 'lmDictCnt', 'lmNegCnt',
                   'lmPosCnt', 'lmStrongCnt', 'lmModCnt', 'lmWeakCnt', 'lmUncerCnt',
                   'lmLitigCnt', 'harvNegCnt']
        df = pd.DataFrame(columns=columns, index=dates)

        for line in Lines[1:]:

            # There is some jank, some of the entries are in double quotes and some are not.
            line = line.strip()
            split = line.split('" ')

            # Assuming that the date filed is for the previous quarter. This may not be the case if a company had to
            # refile for some reason. Also this is where the jank happens since the date filed and the accession number
            # in one single entry in the split list. So we have to split it again.
            split_jank = split[4].split()

            DateFiled = split_jank[0]
            AccessionNumber = split_jank[1].replace('"', '')

            # The rest of the data
            split_data = split[5].split()
            for i in range(len(split_data)):
                split_data[i] = int(split_data[i])
            # Note we are subtracting three months off of the date filed so that the index will correspond to the
            # correct 10Q/10K filing for the respective quarter (for example 10Q for Q1 will be released in Q2).
            tempQDate = (pd.to_datetime(DateFiled, format='%Y/%m/%d') - pd.Timedelta(90, 'D')).to_period('Q')

            df.at[tempQDate, "CIK"] = split[1].replace('"', '')
            df.at[tempQDate, "CompanyName"] = split[2].replace('"', '')
            df.at[tempQDate, "FormType"] = split[3].replace('"', '')
            df.at[tempQDate, "DateFiled"] = DateFiled
            df.at[tempQDate, "AccessionNumber"] = AccessionNumber
            df.at[tempQDate, columns[5:17]] = split_data

        # Can't save a file if it is a DOS file name
        Dos = ["CON", "AUX", "PRN", "LST", "NUL"]
        if file[:-4] not in Dos:
            df.to_pickle(os.path.join(pathOut, file[:-4] + ".pk"))
