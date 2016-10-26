#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import sys
import subprocess
import xlrd
import re
from preprocessor import tweet_preprocessor as tp
reload(sys)
sys.setdefaultencoding('utf-8')
import numpy as np
concept_pattern = r'([^\s\w\,]|_)+'

#tester_tweet_text = "this hadn't a Aspirine rooooooot loooo deperession is knockin US\'s me on :) my assss....#outofit #dead 😴😑 http://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path"
supp_list = ['st johns wort', 'stjohns wort', 'stjohn s wort', 'st john s wort', 'stjohn swort', 'stjohnswort', 'st johnswort' ]

# book = xlrd.open_workbook('../public_sample/DietarySupplementYesSet.xlsx')
book = xlrd.open_workbook('data/raw_data/tweetsy_yes_no_suppl.xlsx')
sheet = book.sheet_by_name("Sheet1")
future_tweet_clus = []
tweets = ""
# Code to run on each tweet.
for row_index in xrange(sheet.nrows):
  col_values = sheet.row(0)
  if row_index >= 1:
    # Scrape only NO tweets.
    # if sheet.cell(row_index, 0).value == 'No':
    	current_tweet_text = sheet.cell(row_index, 1).value
    	icol = 2
    	#current_tweet_id = sheet.cell(row_index, 0).value
    	annotation = sheet.cell(row_index, 0).value
    	if str(current_tweet_text) is not None:
    	  #print current_tweet_text
    	  current_tweet = tp.tweet_preprocessor(current_tweet_text)
    	  tweet = current_tweet.preprocessor().lower()
          big_regex = re.compile('|'.join(map(re.escape, supp_list)))
          tweet = big_regex.sub("stjohnswort", tweet)
          future_tweet_clus.append(tweet + ',' + annotation)
    	  tweets += tweet + " "

np.savetxt('tweetsy_yes_no_suppl_tweet_clustering.csv', future_tweet_clus, delimiter=",", fmt="%s")

text_file = open("tweetsy_yes_no_suppl.txt", "w+")
text_file.write(tweets)
text_file.close()
