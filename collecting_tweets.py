import tweepy,csv,re
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup

consumerKey = ''
consumerSecret = ''
accessToken = ''
accessTokenSecret = ''
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))

def cleanTweet(tweet):
    soup = BeautifulSoup(tweet, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

search_words = "covid vaccine"
date_since = "2020-08-01"

results = [status._json for status in tweepy.Cursor(api.search, 
                                                    q=search_words, 
                                                    count=100, 
                                                    tweet_mode='extended', 
                                                    lang='en').items(3000)]

# Now you can iterate over 'results' and store the complete message from each tweet.
my_tweets = []
for status in results:
    if (status['retweeted'] == False):  # Check if Retweet
        my_tweets.append(status['full_text'])
    else:
        my_tweets.append(status['extended_tweet']['retweeted_stats']['full_text'])
   
import csv

with open('non_cleaned_1.csv', 'w+', encoding='utf-8', newline ='') as csvfile:
    writer = csv.writer(csvfile)
    for tweet in my_tweets:
        writer.writerow([tweet])     
   
test_result = []
for tweet in my_tweets:
    test_result.append(cleanTweet(tweet))
    
# Open/create a file to append data to
clean_df = pd.DataFrame(test_result,columns=['text'])
clean_df.to_csv('clean_tweets_test.csv',encoding='utf-8')
