from __future__ import division
import argparse, os, sys
import tweepy
from textblob import TextBlob
from datetime import datetime
import xlrd

reload(sys)
sys.setdefaultencoding('utf-8')

"""
FeatureCollector connects to Twitter API and collects features from tweets given their IDs
"""
class FeatureCollector:
    def __init__(self, file_preprocessed_path):
        self.file_preprocessed_path = file_preprocessed_path

        # Set file_out_path - add appendinx to input file path
        file_preprocessed_path, file_preprocessed_extension = os.path.splitext(self.file_preprocessed_path)
        self.file_out_path = file_preprocessed_path + '-features_collected' + file_preprocessed_extension

        # Set tweepy
        consumer_key = "yHPzMZDiXTAcGrps4rYTsDqJx"
        consumer_secret = "DpCgO5BCOle530WL8w7HhtbDxNtGvaIk7mdbL8sQ7kDzrLMgsT"
        access_token = "1002228724538990592-78uN64HWD2derk9QTLeYXXHHtExpLt"
        access_token_secret = "vqUNWp0QB3ExZj2UB4VBRDQlO54CpQ463MN1FHo1O6YGc"

        # Stream Twitter API
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(auth)

    def collect_features(self):
        # Open input and output files
        file_in = open(self.file_preprocessed_path, 'r')
        file_out = open(self.file_out_path, 'w')

        # Iterate over input file lines, collect features with get_features
        for sample in file_in.read().splitlines():
            tweet_id, label = sample.split(',')
            features = self.get_features(tweet_id)
            # Write data in csv format
            features_string = ','.join(map(str, features))
            all_data_string = (tweet_id, features_string, label)
            if features_string:
                file_out.write(','.join(all_data_string) + '\n')
            # Skip if no features collected
            else:
                pass
        
        return self.file_out_path

    def get_features(self, tweet_id):
        features = []
        try:
            tweet = self.api.get_status(tweet_id)
            tweet_text = tweet.text.encode('utf-8')
            
            # Get all features
            hashtags_count = len(tweet.entities.get('hashtags'))
            urls_count = len(tweet.entities.get('urls'))
            symbols_count = len(tweet.entities.get('symbols'))
            user_mentions_count = len(tweet.entities.get('user_mentions'))
            letters_count = sum(1 for letter in tweet_text)        
            upper_letters_sum = sum(1 for letter in tweet_text if letter.isupper())
            try:
                ratio_letters_upperletters = round(letters_count/upper_letters_sum, 2)
            except ZeroDivisionError:
                ratio_letters_upperletters = 0
            polarity = round(TextBlob(tweet_text).sentiment.polarity, 2)
            subjectivity = round(TextBlob(tweet_text).sentiment.subjectivity, 2)
            retweets_count = tweet.retweet_count
            favourite_count = tweet.favorite_count
            friends_count = tweet.user.friends_count
            followers_count = tweet.user.followers_count
            try:
                ratio_followers_friends = round(friends_count/followers_count, 2)
            except ZeroDivisionError:
                ratio_followers_friends = 0
            total_posts_count = tweet.user.statuses_count
            verified_user = int(tweet.user.verified)
            account_age = (datetime.now() - (tweet.user.created_at)).days

            hatebase_data = open('hatebase_data.csv', 'r')
            hate_word_contained = 0
            offensiveness_score = 0
            for line in hatebase_data:
                line = line[:-1]
                hate_word, offensiveness = line.split(',')
                if hate_word in tweet_text:
                    hate_word_contained = 1
                    offensiveness_score = offensiveness
            
            swearword_data = xlrd.open_workbook('swearword_data.xlsx')
            swear_word_contained = 0
            for row in range(swearword_data.sheet_by_index(0).nrows):
                if swearword_data.sheet_by_index(0).cell_value(row, 0) in tweet_text:
                        swear_word_contained = 1

            features = [swear_word_contained, hate_word_contained, 
                        offensiveness_score, hashtags_count, urls_count, 
                        symbols_count, user_mentions_count, letters_count, 
                        upper_letters_sum, ratio_letters_upperletters, 
                        polarity, subjectivity, retweets_count, favourite_count, 
                        friends_count,followers_count, ratio_followers_friends, 
                        total_posts_count, verified_user, account_age]

        except tweepy.TweepError:
            pass
        
        return features