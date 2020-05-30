library(rvest)
library(stringi)
library(V8)

# This script will scrape Zacks Earnings for earnings announcements.
#
# Requires that there exists a text file "list_cik.txt" which contains a tab delimited file of the company ticker,
# and CIK number. For example:
# AAPL 0000320193
# The GVkey and CIK are not needed for this and can be replaced with something else or left blank.

fileName <- "list_cik.txt"
conn <- file(fileName,open="r")
linn <-readLines(conn)

path_out <- "DirtyZacksEarnings/"

for (i in 1:length(linn)){

  	split <- strsplit(linn[i],"\t")

	url_1 <- "https://www.zacks.com/stock/research/"
	tkr <- split[[1]][1]
	print(tkr)
	url_2 <- "/earnings-announcements"
	fullString <- paste(url_1,tkr,url_2, sep="")

	print(fullString)
    good <- TRUE
	ctx <- v8()
	tryCatch(pg <- read_html(fullString), error = function(e) {good <<- FALSE})

    if (good) {

	    html_nodes(pg, xpath=".//script[contains(., 'obj_data')]") %>%
  	    html_text() %>%
  	    stri_replace_all_fixed('document.', '') %>%
  	    ctx$eval() -> ignore_the_blank_return_value

	    textFileName <- paste(path_out, tkr, ".txt", sep="")

	    good_htlm <- TRUE

	    tryCatch(dat <- ctx$get("obj_data"), error = function(e) {good_htlm <<- FALSE})

	    if (good_htlm) {write.table(dat[1], file = textFileName)}

     }
}

close(conn)
