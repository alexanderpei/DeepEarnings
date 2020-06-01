# DeepEarnings

This is a project which attempts to model whether a company will beat or miss earnings announcements based 
on previous financial data, sentiment analysis, and stock price movement using deep learning. For example, 
will the amount of long-term debt impact next quarters earnings? How about the amount of R&D spent? Do increases 
in volume contain information about upcoming earnings announcements? Deep learning pulls out complex, 
non-linear statistical dependencies in the data between every variable that are otherwise unobservable to humans. 
The end-goal is to develop trading strategies through reinforcement learning to develop a profitable algorithm. 

Feel free to also use these methods to gather earnings or 10Q/10K data. Also please message me with any questions 
or comments :) 

__Current Performance__
Baseline bias is the natural bias in the data for a company to beat earnings.

| Data Features | Model | Baseline Bias | Train Acc | Test Acc | Num Samples | Time Range | 
| --- | --- | --- | --- | --- | --- | --- |
| Compustat IQ Fundamentals Quarterly | 2 Layer NN | 57.50% | 71.11% | 66.91% | 56621 | 2010-2020 |
| OHCLV 30 prior to earnings          | biLSTM     | 56.84% | 58.89% | 58.34% | 81549 | 2010-2020 |
| 10Q/10K Sentiment                   | 5 Layer NN | 57.24% | 59.60% | 58.82% | 52918 | 2010-2020 |
### Looking for collaborators / to-do list
Feel free to reach out to me if you're interested in collaborating. I'm looking for people who are experienced 
traders who might know how to develop trading strategies off of this. I'm also looking for people who have an
 interest in machine learning with finance applications.
Future ideas:
1. ~~Use LSTMs/RNNs to track OHLCLV + technical indicators before earnings (I've sorta tried this and it sorta works,
need to clean it up before I commit).~~ Improve LSTM hyperparameters and add technical indicators
2. Use advanced NLP methods on 10Q/10K datasets
3. Use reinforcement learning to develop trading strategies
4. Use alternative data (Google trends) 

## Getting Started

This project uses the R package "edgar" to scrape SEC 10Q/10K filings for sentiment analysis. 
You also need basic things like Pandas, scikit, and numpy. The neural networks will be built using Keras. 
I'm using PyCharm for everything so just download that and you can clone the repo. 
Also PyCharm will let you run R scripts. 

You're going to need a text file "list_cik.txt" which contains a list of company tickers and CIK numbers.
This list will provide the companies to the get_ZacksEarnings.R to scrape. If you don't plan on doing sentiment
 analysis, you don't need the CIK numbers. 

## Gathering the data

### Scraping zacks.com earnings

Because I couldn't find access to consensus estimates for earnings easily, I scraped zacks.com for their earnings. 
For example, data are scraped from this link:
https://www.zacks.com/stock/research/AAPL/earnings-announcements

Run this in your terminal:
```
Rscript get_ZacksEarnings.R
```
This will create a directory "DirtyZacksEarnings" where it stores each earnings for a company in a text file.

### Financial data

I have access to Compustat North American Wharton Research Data Service through my university. I don't think I'm allowed to share that data 
set because of their terms and conditions. If you do have access yourself, you can get this data set by querying 
Compustat - Capital IQ / Compustat / North America
Daily / Fundamentals Quarterly through Wharton Research Data Services. For the variables, select all of them.

If you don't have access to this data set, I recommend this free
API https://financialmodelingprep.com/developer/docs/ I'm not sure how reliable this website is, but it's hard to
come by free financial data. IEXCloud is probably another good choice, but it is not free after you hit a certain
number of requests. Also if you're a beast, you could manually get this data directly from the 10Q/10K documents
themselves. If you do end up using an alternative method, then you'll have customize your own script to clean
the data you gather.

This is the main chunk of data upon which the algorithm is trained on. The algorithm will receive data from balance 
sheets, cash flow, debt, and much more, from the previous quarter. From these data, the algorithm will try to 
predict if the company will beat earnings or not. 

### CRSP stock data

I have access to CRSP Stock Prices Wharton Research Data Service through my university. There are 
alternatives like pyfinance to get daily stock prices. You can replicate my query by doing:
CRSP / Quarterly UpdateStock / Security Files / CRSP Daily Stock. For the desired variables:
Company Name, Ticker, Price, Share Volume, Open Price, Ask or High, Bid or Low, Closing Bid, 
Closing Ask, Return without Dividends.

### Downloading 10Q/10K files and sentiment analysis 

Again this will require the text file "list_cik.txt". I'm using the R package "edgar", which to me as been the 
easiest way to scrap SEC filings. This package requires a CIK number to scrape filings from. The current 
sentiment analysis built in to the edgar package is very simple. It just counts the number of positive/negative 
business words. More advanced NLP methods would obviously be better, but I haven't gotten around to that.
``` 
Rscript get_SecSentiment.R
```

## Cleaning the Data

The data is converted to a Pandas data frame which has dates as the indexes. The time ranges are hardcoded to only 
be from 01/2008 - 01/2021 so if you're using data before hand or you're a time traveler then you need to change 
those in the script.

### Clean zacks.com earnings

After you have run the R script which gathers the earnings from zacks.com, you can now run the script to clean 
these text files into a Pandas data frame:
```
python pp_CleanZacksEarnings.py
```
This will make a data frame formatted like so. These are a few earnings from AAPL:

| Quarter (index for the df) | DateAnnounced | Estimate | Reported | Surprise | pctSurprise | AMC |
| --- | --- | --- | --- | --- | --- | --- |
| 2019Q2 | 2019-07-30 00:00:00 | 2.1  | 2.18 | 0.08 | 3.80952 | True |
| 2019Q3 | 2019-10-30 00:00:00 | 2.84 | 3.03 | 0.19 | 6.69014 | True |
| 2019Q4 | 2020-01-28 00:00:00 | 4.54 | 4.99 | 0.45 | 9.91189 | True |
| 2020Q1 | 2020-04-30 00:00:00 | 2.09 | 2.55 | 0.46 | 22.0096 | True |

### Clean Compustat data

The queried compustat data is a long tab delimited text file with all of the dates and companies 
stacked on top of each other. 

```
python pp_CleanCompustat.py
```
This will make a data frame formatted like so. These are a few entries for the Compustat Fundamentals Quarterly for 
AAPL. For example, mkvaltq is the market capitalization.

|Quarter (index for the df) | gvkey | datadate | fyearq | fqtr |  fyr | indfmt | ... | mkvaltq | prccq | prchq | prclq | adjex | spcseccd | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
|2019Q2 | 001690 | 20190630 | 2019 | 3 | 9 | INDL | ... | 896853.6984  | 197.9200 | 215.3100 | 170.2700 | 1.0000 | 940 | 
|2019Q3 | 001690 | 20190930 | 2019 | 4 | 9 | INDL | ... | 995151.5669  | 223.9700 | 226.4200 | 192.5800 | 1.0000 | 940 | 
|2019Q4 | 001690 | 20191231 | 2020 | 1 | 9 | INDL | ... | 1287643.2104 | 293.6500 | 293.9700 | 215.1320 | 1.0000 | 940 | 
|2020Q1 | 001690 | 20200331 | 2020 | 2 | 9 | INDL | ... | 1099546.6542 | 254.2900 | 327.8500 | 212.6100 | 1.0000 | 940 | 

### Clean CRSP stock data 

Similar process to cleaning the Compustat data. 

``` 
python pp_CLeanStockPrice.py
```

| Date | permno  | date | ticker | comnam | bidlo | ... | vol | bid | ask | openprc | retx |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|2010-01-05 | 14593 | 20100105 | AAPL | APPLE INC | 213.25000 | ... | 22082278 | 214.35001 | 214.38000 | 214.60001 | 0.001729  |
|2010-01-06 | 14593 | 20100106 | AAPL | APPLE INC | 210.75000 | ... | 20396105 | 210.92999 | 210.94000 | 214.38000 | -0.015906 |
|2010-01-07 | 14593 | 20100107 | AAPL | APPLE INC | 209.05000 | ... | 17628746 | 210.53999 | 210.52000 | 211.75000 | -0.001849 |
|2010-01-08 | 14593 | 20100108 | AAPL | APPLE INC | 209.06000 | ... | 16553861 | 211.96001 | 212.00000 | 210.30000 | 0.006648  |
|2010-01-09 |   NaN |      NaN |  NaN |       NaN |      NaN  | ... |     NaN  |      NaN  |      NaN  |      NaN  |      NaN  |
|2010-01-10 |   NaN |      NaN |  NaN |       NaN |      NaN  | ... |     NaN  |      NaN  |      NaN  |      NaN  |      NaN  |
|2010-01-11 | 14593 | 20100111 | AAPL | APPLE INC | 208.45000 | ... | 17085997 | 210.02000 | 210.13000 | 212.80000 | -0.008822 |

### Clean sentiment data 
Same thing as above, must have ran the get_SecSentiment.R script before this one.
``` 
python pp_CleanSecSentiment.py
```
This will make a data frame formatted like so. These are a few entries for the sentiment analysis
for AAPL. For example, lmWeakCnt are the number of Loughran-McDonald weak words in the 10Q/10K document.

| Quarter (index for the df) | CIK | CompanyName | FormType | DateFiled | ... | lmWeakCnt | lmUncerCnt | lmLitigCnt | harvNegCnt |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| 2019Q2 | 320193 | Apple Inc | 10-Q | 2019-07-31 | ... | 157 | 367 | 236 | 1356 |
| 2019Q3 | 320193 | Apple Inc | 10-K | 2019-10-31 | ... | 177 | 438 | 265 | 1666 |
| 2019Q4 | 320193 | Apple Inc | 10-Q | 2020-01-29 | ... | 44  | 147 | 93  | 641  |
| 2020Q1 | 320193 | Apple Inc | 10-Q | 2020-05-01 | ... | 47  | 153 | 97  | 831  |

## Training the networks

This current iteration does not incorporate the sentiment analysis. I've tried it and it does not increase 
accuracy substantially, however I will be adding it back in the future for completeness.

### Multilayer perceptron for compustat quarterly financial data

__Preprocessing__

Financial data is messy because comapanies may omit certain financial data fields (Apple may not have reported their
good will for example). KNN imputation is a method used to fill in missing nan values based on closer points according
to some distance metric. This may or may not be a valid assumption for certain data fields. 

The data are also scaled by the market capitalization of every company to allow fair comparisons between companies'
financial data.

The data are also whitened (zero mean and unit variance across features) which can improve neural network training.

__Training__

The data set contains 56621 samples after cleaning. Train:test:validation set are split 0.8:0.1:0.1. 

The network will contain two hidden layers with 250 hidden units each implented using Keras. SeLU activations were 
chosen opposed to ReLU for better performance. l2 regularization and dropout layers also reduced overfitting and 
improved test accuracy.


![nn](./pics/nn-plot.png)

### LSTM for OHLCV data 

__Preprocessing__

Most stock price data is clean. 30 trading days worth of data before the announced earnings date are used as features 
for the LSTM. Each feature is normalized across time within each sample. Otherwise Volume would be several
orders of magnitude higher than every other feature). This way, The LSTM will be forced to learn statistics about the
dynamics of the OHLCV movement. 

Additionally, "background" market data in the form of SPY will be added for each time frame. This may give an indication
of how the market is doing in general.

__Training__

A single biLSTM layer with 10 hidden units is used along with a dropout layer to reduce overfitting.

Future directions will be to add more technical indicators. As you can see from the plots, the accuracy isn't too great,
and the parameters definitely need tuning. Including the background SPY data definitely helps.

![nn](./pics/lstm-plot.png)

### Multilayer perceptron for sentiment data 

__Preprocessing__ 

All of the LM word counts were divided by the total number of words in the 10Q/10K document to create a 
frequency for the occurence of certain word categories. The samples were normalized across features
so that each feature had zero mean and unit variance for improved neural network training.

__Training__

SeLU layers were doing some weird stuff where the validation accuracy was higher than the training accuracy.
5 ReLU layers were used instead. Also RMSprop over Adam made nicer looking training plots.

![nn](./pics/senti-plot.png)