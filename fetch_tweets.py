# coding: utf-8

# In[ ]:


from twython import TwythonStreamer
from collections import Counter
from termcolor import *
import csv
import pandas as pd
import os
import sys

APP_KEY = XXXX
APP_SECRET = XXXX
OAUTH_TOKEN = XXXX
OAUTH_TOKEN_SECRET = XXXX

os.chdir('D:/project026')
MaxTweetsAPI = 200
wait_on_rate_limit = True

counter = Counter()
counter = 0
header_done = False
file_name_suffix = 0

# enter number of tweets to write to csv file
keep_tweets = int(raw_input("How many tweets do you want to keep? (-1 for unlimited): "))
tweets_per_file = int(raw_input("How many tweets do you want to save per file? "))

class MyStreamer(TwythonStreamer):

    def on_success(self,data):
        global counter
        global keep_tweets
        global tweets_per_file
        global file_name_suffix
        counter += 1
        global header_done
         

        if counter < keep_tweets:
            if 'text' in data:
                if not data['retweeted'] and "RT @" not in data['text']:
                    IDs = data['user']['id']
                    Location = data['user']['location']
                    Tweet = data['text'].encode('utf-8')
                    info = [IDs, Location, Tweet]
                    print info
                    print counter
                        
                    with open("immuno_" + str(file_name_suffix) + ".csv", 'ab+') as tweets:
                        
                        if counter % 1000 == 0:
                            print counter, "tweets processed"

                        # Increment the file name 
                        if counter % tweets_per_file == 0:
                            file_name_suffix += 1
                            header_done = True # reenable if you want every file to include the header
                        
                        if not header_done:
                            header = csv.DictWriter(tweets, fieldnames=['ID','Location','Tweet'])
                            header.writeheader()
                            print header
                            header_done = True   
                        
                        writer = csv.writer(tweets)
                        ID = data['user']['id']
                        location = data['user']['location']
                        text = data['text'].encode('utf-8')
                        coords = data['coordinates']
                        tweeties = ([ID], [location], [text])
                        stopWords = (['''#,u',xa,@,-,_,/,\,?,|,/,!,:,;,(,$,),[,],}'''])
                        words = text.split()

                        if words not in stopWords:
                            writer.writerow([tweeties])
                            print tweets
                        
        if counter >= keep_tweets:
            counter += 1
            self.disconnect()
            tweets.close()
            print ("Finito!")
            return False

    def on_error(self,status_code,data):
        print ("not again :-(")
        print (status_code,data )
        
if __name__ == '__main__':

    print(colored(
        """Please note that streaming tweets may take longer than other programs because this one is 
        more awesome...and because of pre-processing of tweet data""",
        'blue'))
    print(colored("Twitter stream initializing...",'green'))
    stream = MyStreamer(APP_KEY,APP_SECRET,OAUTH_TOKEN,OAUTH_TOKEN_SECRET)
    stream.statuses.filter(
        track=["cancer immunotherapy","immuno-oncology","immunooncology","cancer biotherapy",
               "cancer immunology","onco-immunology", "immune checkpoint drug","cancer vaccine", 
               "personalized cell therapy", "immune checkpoint blocker", "immune checkpoint inhibitor", 
               "adoptive T cell therapy"])
    results = stream.statuses.filter()
    print results

