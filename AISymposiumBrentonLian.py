import praw  # Reddit API
import pandas as pd
import re
from dotenv import load_dotenv
import os

# Reddit developer account and enviromental variables
load_dotenv()

client_id = os.getenv("REDDIT_CLIENT_ID")
client_secret = os.getenv("REDDIT_CLIENT_SECRET")
user_agent = os.getenv("REDDIT_USER_AGENT")

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# Fetch data for all hot submissions in the "UNC" subreddit
submissions_data = []
for submission in reddit.subreddit("UNC").hot(limit=None):
    submissions_data.append({
        'Title': submission.title,
        'ID': submission.id,
        'Author': submission.author,
        'Created_UTC': submission.created_utc,
        'Score': submission.score,
        'Upvote_Ratio': submission.upvote_ratio,
        'URL': submission.url
    })

# Create a DataFrame from the collected data
unc_df = pd.DataFrame(submissions_data)

# Display the first few rows of the DataFrame
#print(unc_df.head())
#print(unc_df["Title"].head())

#Create a function to clean the tweets
def cleanTxt(text):
 text = re.sub(r'@[A-Za-z0-9]+', '', text) #Remove @mentions replace with blank
 text = re.sub(r'#', '', text) #Remove the ‘#’ symbol, replace with blank
 text = re.sub(r'RT[\s]+', '', text) #Removing RT, replace with blank
 text = re.sub(r'https?:\/\/\S+', '', text) #Remove the hyperlinks
 text = re.sub(r':', '', text) # Remove :
 return text

#Cleaning the text
unc_df["Title"]= unc_df["Title"].apply(cleanTxt)

#Show the clean text
#unc_df.head()

#Define the function to remove emjois and Unicode characters
def remove_emoji(string):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
    "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', string)

#Cleaning the text
unc_df["Title"] = unc_df["Title"].apply(remove_emoji)

#Show the clean text
#unc_df.head()

#import sentiment analysis libraries
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create new columns for subjectivity and polarity and add them to the reddit_df DataFrame
unc_df['Subjectivity'] = unc_df['Title'].apply(getSubjectivity)
unc_df['Polarity'] = unc_df['Title'].apply(getPolarity)

# Display the data
#print(unc_df.head())

# Define the function to categorize polarity into different insights
def getInsight(score):
    if score < 0:
        return "Negative"
    elif score == 0:
        return "Neutral"
    else:
        return "Positive"

# Apply the function to create a new 'Insight' column in the unc_df DataFrame
unc_df["Insight"] = unc_df["Polarity"].apply(getInsight)

# Display the first 50 rows with the new 'Insight' column
#print(unc_df[['Title', 'Polarity', 'Insight']].head(50))

#visualization imports
import seaborn as sns
import warnings
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Sentiment graph
plt.title("UNC sentiment scores")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.rcParams["figure.figsize"] = (10, 8)
unc_df["Insight"].value_counts().plot(kind="bar", color="#2078B4")
plt.show()

stopwords = STOPWORDS
#print(stopwords)
#Let checkout the stop words in Pythonfrom wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Combine all titles into a single string
text = ' '.join([twts for twts in unc_df['Title']])

# Generate word cloud
wordcloud = WordCloud(
    width=1000,
    height=600,
    max_words=100,
    stopwords=STOPWORDS,
    background_color="black",
    collocations=False  # Avoid displaying frequently co-occurring words
).generate(text)

# Display the generated image:
plt.figure(figsize=(20, 10), facecolor='k')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

#Print positive title
print("Most positive posts according to textblob")
# Sort the DataFrame by 'Polarity' column in descending order
sorted_df_positive = unc_df.sort_values(by='Polarity', ascending=False)

# Print the top 30 rows
print(sorted_df_positive[['Title', 'Polarity', 'Insight']].head(30))

#print Negative title
print()
print()
print("Most negative posts according to textblob")
# Sort the DataFrame by 'Polarity' column in ascending order
sorted_df_negative = unc_df.sort_values(by='Polarity', ascending=True)

# Print the top 30 rows
print(sorted_df_negative[['Title', 'Polarity', 'Insight']].head(30))

print(unc_df.shape)

all_columns = unc_df.columns.tolist()
print(all_columns)

#Print most subjective title
print("Most subjective posts according to textblob")
# Sort the DataFrame by 'Polarity' column in descending order
sorted_df_positive = unc_df.sort_values(by='Subjectivity', ascending=False)

# Print the top 30 rows
print(sorted_df_positive[['Title', 'Subjectivity', 'Insight']].head(30))

#print least subjective title
print()
print()
print("Least subjective posts according to textblob")
# Sort the DataFrame by 'Polarity' column in ascending order
sorted_df_negative = unc_df.sort_values(by='Subjectivity', ascending=True)

# Print the top 30 rows
print(sorted_df_negative[['Title', 'Subjectivity', 'Insight']].head(30))

