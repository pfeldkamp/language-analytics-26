# %%

from utils import load_and_concat

import pandas as pd
from datasets import load_dataset
import spacy
nlp = spacy.load("en_core_web_md")
import matplotlib.pyplot as plt
import seaborn as sns

# sentiment modules
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer


# %%
# try out the sentiment dictionary
sia = SentimentIntensityAnalyzer()

text = "I absolutely love Columbo. They almost get away and then 'just one more thing'. So sad I can't just take 2 weeks, rewatch the whole thing."
scores = sia.polarity_scores(text)
print(scores)

# sentence tokenization
doc = nlp(text)
sentences = list(doc.sents)
scores = [sia.polarity_scores(str(s))['compound'] for s in sentences]
print(f"{len(sentences)} sentences:", scores)
print("mean: ", sum(scores)/len(scores))
# %%

##### With Dataset #####

# load our dataset
df = load_and_concat("cornell-movie-review-data/rotten_tomatoes")
df.head()
# %%

# score all texts
df['scores'] = [sia.polarity_scores(str(s))['compound'] for s in df['text']]
df.head()


# %%
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 1]['scores'], color='blue', label='1', kde=True, stat='density')
sns.histplot(df[df['label'] == 0]['scores'], color='orange', label='0', kde=True, stat='density')
plt.title('Scores across groups')
plt.xlabel('Compound score')
plt.ylabel('Density')
plt.legend()
plt.show()

# %%

# remove 0s
filtered_df = df.loc[df['scores'] != 0]

plt.figure(figsize=(10, 6))
sns.histplot(filtered_df[filtered_df['label'] == 1]['scores'], color='blue', label='1', kde=True, stat='density')
sns.histplot(filtered_df[filtered_df['label'] == 0]['scores'], color='orange', label='0', kde=True, stat='density')
plt.title('Scores across groups')
plt.xlabel('Compound score, filtered')
plt.ylabel('Density')
plt.legend()
plt.show()

# %%
subset = filtered_df[:5000]
averages = []

for text in subset['text']:
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = [sia.polarity_scores(str(s))['compound'] for s in sentences]
    avg_score = sum(scores) / len(scores)
    averages.append(avg_score)

subset['avg_scores'] = averages

# %%

plt.figure(figsize=(10, 6))
sns.histplot(subset[subset['label'] == 1]['avg_scores'], color='blue', label='1', kde=True, stat='density')
sns.histplot(subset[subset['label'] == 0]['avg_scores'], color='orange', label='0', kde=True, stat='density')
plt.title('Scores across groups')
plt.xlabel('Compound score, filtered')
plt.ylabel('Density')
plt.legend()
plt.show()


# %%

# CAVEATS to old-school dictionary way

data = {"irony": "Oh great, another Monday.",
        "negation": "It's not bad.",
        "context_dependent_words": "That performance was sick.",
        "understatement": "I’ve seen worse."}

for key, value in data.items():
    score = sia.polarity_scores(value)["compound"]
    print(f"{key:<25} {score:>8.3f}  {value}")
# %%
