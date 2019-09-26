import csv
from example import *
import time
import pandas as pd

data = pd.read_csv("dfe.csv",encoding = "ISO-8859-1")

CONSUMER_KEY = 'n4qNDyH25RLTRW1wLlKa4lLU6'
CONSUMER_SECRET = 'OIN2Vi1Wdq8mP8QLFaiN6sGvc2LeVHLdJMECVEiyrUk9j8z9zD'
ACCESS_TOKEN = '1171636288086315009-3uesxkcQ5TKbijuUaITb05yOr8fu51'
ACCESS_TOKEN_SECRET = 'aY654202Zk9LCJ01x9zcTfqarMpbybp5we5cbKUDZQxEG'
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)



def followercount(names):
    for name in names:
        try:
            user = api.get_user(name)
        except:
            return 0
        # ageTwitterAccount = user.created_at
        followers_count =  user.followers_count
        timezone = user.time_zone
        print(name)
        print(ageTwitterAccount)
        print(followers_count)
        print(friends_count)
        print(timezone)
        return followers_count


followerlist = []
name = []


for i in range(0, 5):
    name.append(data.name[i + 1])
(', '.join('"' + i + '"' for i in name))
# print(name)

for i in range(0, len(name)):
    followers = followercount(name)
    followerlist.append(followers)

with open('followers.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    filewriter.writerow(['Name', 'followers'])
    for i in range(len(name)):
        filewriter.writerow([name[i], followerlist[i]])
