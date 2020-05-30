# DeepEarnings

This is a project which attempts to model whether a company will beat or miss earnings announcements based on previous financial data, sentiment analysis, and stock price movement using deep learning. For example, will the amount of long-term debt impact next quarters earnings? How about the amount of R&D spent? Do increases in volume contain information about upcoming earnings announcements? Deep learning pulls out complex, non-linear statistical dependencies in the data between every variable that are otherwise unobservable to humans. The end-goal is to develop trading strategies through reinforcement learning to develop a profitable algorithm. 

Feel free to also use these methods to gather earnings or 10Q/10K data. Also please message me with any questions or comments :) 

Current model test accuracy: 66.91%

Current model train accuracy: 71.11% 

Baseline: 57.5% bias in classes to beat earnings

Number of data points: 56621

Timerange: 2010 - 2020

### Looking for collaborators / to-do list
Feel free to reach out to me if you're interested in collaborating. I'm looking for people who are experienced traders who might know how to develop trading strategies off of this. I'm also looking for people who have an interest in machine learning with finance applications.
Future ideas:
1. Use LSTMs/RNNs to track OHLCLV + technical indicators before earnings (I've sorta tried this and it sorta works, need to clean it up before I commit).
2. Use advanced NLP methods on 10Q/10K datasets
3. Use reinforcement learning to develop trading strategies
4. Use alternative data (Google trends) 

## Getting Started

This project uses the R package "edgar" to scrape SEC 10Q/10K filings for sentiment analysis. You also need basic things like Pandas, scikit, and numpy. The neural networks will be built using Keras. I'm using PyCharm for everything so just download that and you can clone the repo. Also PyCharm will let you run R scripts. 

You're going to need a text file "list_gvkey_cik.txt" which contains a list of company tickers, GVkeys, and CIK numbers. This list will provide the companies to the get_ZacksEarnings.R to scrape. If you don't plan on doing sentiment analysis, you don't need the CIK numbers. The GVkeys will also be used later for the stock movement data, which I haven't gotten to yet.

## Gathering the Data

### Scraping zacks.com earnings

Because I couldn't find access to consensus estimates for earnings easily, I scraped zacks.com for their earnings. For example, data are scraped from this link:
https://www.zacks.com/stock/research/AAPL/earnings-announcements

Run this in your terminal:
```
Rscript get_ZacksEarnings.R
```
This will create a directory "DirtyZacksEarnings" where it stores each earnings for a company in a text file.

### Financial Data

I have access to Compustat North American through my university. I don't think I'm allowed to share that data set because of their terms and conditions. If you do have access yourself, you can get this data set by querying Compustat - Capital IQ / Compustat / North America
Daily / Fundamentals Quarterly through Wharton Research Data Services. For the variables, select all of them.

This is the main chunk of data upon which the algorithm is trained on. The algorithm will receive data from balance sheets, cash flow, debt, and much more, from the previous quarter. From these data, the algorithm will try to predict if the company will beat earnings or not. 

### Downloading 10Q/10K files and sentiment analysis (optional)

Again this will require the text file "list_gvkey_cik.txt". I'm using the R package "edgar", which to me as been the easiest way to scrap SEC filings. This package requires a CIK number to scrape filings from. The current sentiment analysis built in to the edgar package is very simple. It just counts the number of positive/negative business words. More advanced NLP methods would obviously be better, but I haven't gotten around to that.
``` 
Rscript get_SecSentiment.R
```

## Cleaning the Data

The data is converted to a Pandas data frame which has dates as the indexes. The time ranges are hardcoded to only be from 01/2008 - 01/2021 so if you're using data before hand or you're a time traveler then you need to change those in the script.

### Clean zacks.com earnings

After you have run the R script which gathers the earnings from zacks.com, you can now run the script to clean these text files into a Pandas data frame:
```
python pp_CleanZacksEarnings.py
```

### Clean compustat data
This script will save an X and y text files containing the training data with the labels. Rows correspond to samples while columns correspond to features. 

```
python pp_CleanCompustat.py
```

### Clean sentiment data (optional)
Same thing as above, must have ran the get_SecSentiment.R script before this one.
``` 
python pp_CleanSecSentiment.py
```

## Training the Network

This current iteration does not incorporate the sentiment analysis. I've tried it and it does not increase accuracy substantially.

### Multilayer perceptron for compustat quarterly financial data

Companies beat earnings 61% of the time. To account for this bias in the data, the classes were balanced such that 50% of the labels are beat earnings and 50% are miss. The network will contain four hidden layers with 250 hidden units each implented using Keras. SeLU activations were chosen opposed to ReLU for better test data. l2 regularization also reduced over fitting and improved test accuracy.
