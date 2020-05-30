library(XML)
library(RCurl)
library(edgar)
library(rvest)
library(stringi)
library(V8)

# This script will use the R package edgar to download sentiment analysis from the SEC 10Q/10K filings. This will also
# create a few directories to store the download filings. This will probably take forever if you do it for multiple
# companies and for multiple time periods.
# See the online manual for details. https://cran.r-project.org/web/packages/edgar/index.html
#
# Requires that there exists a text file "list_cik.txt" which contains a tab delimited file of the company ticker,
# and CIK number. For example:
# AAPL 0000320193

path_out <- "DirtySentiment/"
fileName <- "list_cik.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)

for (i in 1:length(linn)){

    split <- strsplit(linn[i],'\t')
	tkr <- split[[1]][1]
	cik <- as.numeric(split[[1]][2])
    print(tkr)
	if (length(cik) > 0) {
        good <- TRUE
        tryCatch(header.df <- getSentiment(cik.no = cik, c('10-K', '10-Q'), filing.year = c(2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020)), error = function(e) {good <<- FALSE})

        if (good) {
	        name_out <- paste(path_out, tkr, ".txt", sep="")
	        write.table(header.df, file=name_out)
        }
	}
}

close(conn)