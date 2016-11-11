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
book = xlrd.open_workbook('data/approved_future_formats/id_text_anno_37000.xlsx')
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
    	#icol = 2
    	current_tweet_id = sheet.cell(row_index, 2).value
    	annotation = sheet.cell(row_index, 0).value


    	if str(current_tweet_text) is not None:
    	  #print current_tweet_text
    	  current_tweet = tp.tweet_preprocessor(current_tweet_text)
    	  tweet = current_tweet.preprocessor().lower()
          big_regex = re.compile('|'.join(map(re.escape, supp_list)))
          tweet = big_regex.sub("stjohnswort", tweet)
          print current_tweet_id
          future_tweet_clus.append(tweet + ',' + annotation + ',' + current_tweet_id)
    	  tweets += tweet + " "

np.savetxt('37k_cleaned_tweets_txt_ann_id_tagger_input.csv', future_tweet_clus, delimiter=",", fmt="%s")

text_file = open("37k_w2v_input.txt", "w+")
text_file.write(tweets)
text_file.close()
