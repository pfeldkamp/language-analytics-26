# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
# %%

org = pd.read_csv("../data/doi-10/NarraDetect_Large.csv")
org.columns = [x.lower() for x in org.columns]
print(len(org))
org.head()

# %%
data = pd.read_csv("../data/doi-10/NarraDetect_Scalar_AllModels.csv")
data.columns = [x.lower() for x in data.columns]
data = data[['id','genre', 'label', 'text']]
# fiction /nonfiction labels to POS/NEG
data.head()
# %%
# merge data with df on id
df = pd.concat([org, data], axis=0, ignore_index=True)
df['label'] = df['label'].apply(lambda x: 'narrative' if x == 'POS' else 'non-narrative')

print(len(df))

# %%
print(df['genre'].value_counts())
print(df['label'].value_counts())
df.tail()

# %%
# sentence tokenize all text
df['sentences'] = df['text'].apply(sent_tokenize)
# show average sentlen per genre
df['sentlen'] = df['sentences'].apply(len)
for genre in df['genre'].unique():
    print(f"{genre}: {df[df['genre'] == genre]['sentlen'].mean()}, {df[df['genre'] == genre]['sentlen'].std()}")
sns.boxplot(x='genre', y='sentlen', data=df)
plt.title('Sentence Length by Genre')
plt.xticks(rotation=45)
plt.show()

# %%
df['tokens'] = df['text'].apply(word_tokenize)
df['types'] = df['tokens'].apply(lambda x: len(set(x)))
df['TTR'] = df['types'] / df['tokens'].apply(len)

for genre in df['genre'].unique():
    print(f"{genre}: {df[df['genre'] == genre]['TTR'].mean()}, {df[df['genre'] == genre]['TTR'].std()}")
sns.boxplot(x='genre', y='TTR', data=df)
plt.title('Type-Token Ratio by Genre')
plt.xticks(rotation=45)
plt.show()
# %%
# legal vs flash histogram of TTR
plt.figure(figsize=(10, 6))
sns.histplot(df[df['genre'] == 'LEGAL']['TTR'], color='blue', label='LEGAL', kde=True, stat='density')
sns.histplot(df[df['genre'] == 'FLASH']['TTR'], color='orange', label='FLASH', kde=True, stat='density')
plt.title('Type-Token Ratio Distribution for LEGAL vs FLASH')
plt.xlabel('Type-Token Ratio')
plt.ylabel('Density')
plt.legend()

# %%
# same for label
plt.figure(figsize=(10, 6))
sns.histplot(df[df['label'] == 'fiction']['TTR'], color='blue', label='Fiction', kde=True, stat='density')
sns.histplot(df[df['label'] == 'nonfiction']['TTR'], color='orange', label='Nonfiction', kde=True, stat='density')
plt.title('Type-Token Ratio Distribution for Fiction vs Nonfiction')
plt.xlabel('Type-Token Ratio')
plt.ylabel('Density')
plt.legend()
# %%
df.loc[df['genre'] == 'SCOTUS']['text'].iloc[2]
# %%
# average token length by genre
df['token_len'] = df['tokens'].apply(lambda x: len(x))
for genre in df['genre'].unique():
    print(f"{genre}: {df[df['genre'] == genre]['token_len'].mean()}, {df[df['genre'] == genre]['token_len'].std()}")
sns.boxplot(x='genre', y='token_len', data=df)
plt.title('Average Token Length by Genre')
plt.xticks(rotation=45)
plt.show()

# %%
