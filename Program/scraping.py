import snscrape.modules.twitter as sntwitter
import pandas as pd

query = "genocide since:2023-10-07 until:2023-10-20 lang:en"
tweets = []

for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i > 1000:  # batas jumlah tweet
        break
    tweets.append([tweet.date, tweet.content, tweet.user.username])

df = pd.DataFrame(tweets, columns=["Date", "Content", "Username"])
df.to_csv("free_palestine.csv", index=False)
