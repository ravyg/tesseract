#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import sys
import config
import subprocess
import xlrd
import re
from preprocessor import tweet_preprocessor as tp
reload(sys)
sys.setdefaultencoding('utf-8')

concept_pattern = r'([^\s\w\,]|_)+'

#tester_tweet_text = "this hadn't a Aspirine rooooooot loooo deperession is knockin US\'s me on :) my assss....#outofit #dead 😴😑 http://stackoverflow.com/questions/279237/import-a-module-from-a-relative-path"

# book = xlrd.open_workbook('../public_sample/DietarySupplementYesSet.xlsx')
book = xlrd.open_workbook('../public_sample/tensorflowfile.xlsx')
sheet = book.sheet_by_name("Sheet1")

tweets = ""
# Code to run on each tweet.
for row_index in xrange(sheet.nrows):
  col_values = sheet.row(0)
  current_tweet_text = sheet.cell(row_index, 1).value
  icol = 2
  current_tweet_id = sheet.cell(row_index, 0).value

  current_tweet = tp.tweet_preprocessor(current_tweet_text)
  tweet = current_tweet.preprocessor()
  tweets += tweet.lower()+" "

print tweets

text_file = open("tensorflowfile.txt", "w+")
text_file.write(tweets)
text_file.close()
