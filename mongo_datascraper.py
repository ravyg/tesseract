#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import os
import sys
import pymongo
from pymongo import MongoClient
# from bson import BSON
from bson import json_util
import json
import time
import re
from preprocessor import tweet_preprocessor as tp
reload(sys)
sys.setdefaultencoding('utf-8')

def mongo_connect():
  try:
    client = MongoClient("mongodb://localhost:27017")
    db = client.tweetsy
    return db
  except:
    print('Error: Unable to Connect')
    return None

# Main Execution begins here.
if __name__ == '__main__':
  db = mongo_connect()
  cols = db.collection_names()
  cleaned_tweets = ""
  for c in cols:
    
    if c is not None and c == "melatonin_copy":
    #if c is not None and c == "echinacea_copy":  
      text_file = open("cleaned_data/"+c+".txt", "w+")
      coll = db[c]
      tweets = coll.find()
      # Get time of execution.
      start_time = time.time()
      print start_time
      for tweet in tweets.batch_size(30):
      #for tweet in tweets:
        # That's a workaround to fix tweet json.
        tweet_json_dump = json.dumps(tweet, sort_keys=True, indent=4, default=json_util.default)
        tweet_json = json.loads(tweet_json_dump)
        tweet_text = tweet_json['text']
        # Cleaning text.
        current_tweet = tp.tweet_preprocessor(tweet_text)
        cleaned_tweet = current_tweet.preprocessor().lower()
        print cleaned_tweet
        cleaned_tweets = cleaned_tweets + " " + cleaned_tweet

        #text_file = open("cleaned_data/"+c+".txt", "w+")
        text_file.write(cleaned_tweet)
      text_file.close()
      print "Saved " + "cleaned_data/"+c+".txt"
  print "Saved cleaned text for all collections"
  print("--- %s seconds ---" % (time.time() - start_time))
  # Run Batched Word2Vec now.






