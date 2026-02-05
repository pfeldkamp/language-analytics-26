# %%
import umap
import pandas as pd
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# %%
df = pd.read_csv("/Users/au324704/Downloads/whole_corpus.csv")
df.head()
# %%
df = df[['id', 'text', 'category', 'author', 'book/prompt']].copy()

# %%

# embed texts
model = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = model.encode(terms)

all_texts = df['text'].tolist()
embeddings = model.encode(all_texts)

# %%

# dim red
reducer = umap.UMAP(
    n_neighbors=6,
    min_dist=0.3,
    metric="cosine",
    random_state=42)

# get coords
coords = reducer.fit_transform(embeddings)

# %%


# add UMAP coords back to df
df["umap_1"] = coords[:, 0]
df["umap_2"] = coords[:, 1]

# only get 3 authors for plotting
df_sampled = df.loc[df['author'].isin(['alcott','dickens','austen'])].copy()

plt.figure(figsize=(15, 9))
sns.set_style("whitegrid")

sns.scatterplot(
    data=df_sampled,
    x="umap_1",
    y="umap_2",
    hue="author",  # color by author
    style="category", # marker shape by authentic / not
    s=60,
    alpha=0.3)

plt.title("UMAP of Text Embeddings: Author (color) × Category (shape)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

# %%

# binary labels
y = df["category"].map({"authentic": 1, "synthetic": 0}).values

# feature matrix
X = embeddings  # shape: (n_samples, 384)

authors = df["author"].values

def one_rf_run(df, X, y, authors, n_per_class=100, test_size=0.2, random_state=None):
    rng = np.random.default_rng(random_state)
    selected_idx = []

    for author in np.unique(authors):
        auth_idx = np.where(
            (authors == author) & (y == 1)
        )[0]
        syn_idx = np.where(
            (authors == author) & (y == 0)
        )[0]

        # sample evenly
        auth_sample = rng.choice(auth_idx, n_per_class, replace=False)
        syn_sample = rng.choice(syn_idx, n_per_class, replace=False)

        selected_idx.extend(auth_sample)
        selected_idx.extend(syn_sample)

    selected_idx = np.array(selected_idx)

    X_sub = X[selected_idx]
    y_sub = y[selected_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sub,
        y_sub,
        test_size=test_size,
        stratify=y_sub,
        random_state=random_state
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }


# now loop 100 times:
results = []

for r in range(100):
    res = one_rf_run(
        df, X, y, authors,
        n_per_class=100,
        random_state=r
    )
    results.append(res)

accs = [r["accuracy"] for r in results]
f1s  = [r["f1"] for r in results]

print("Mean accuracy:", np.mean(accs))
print("Std accuracy:", np.std(accs))
print("Mean F1:", np.mean(f1s))

# %%

cv_accs = []

for author in np.unique(authors):
    train_idx = authors != author
    test_idx  = authors == author

    rf = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X[train_idx], y[train_idx])
    y_pred = rf.predict(X[test_idx])

    acc = accuracy_score(y[test_idx], y_pred)
    cv_accs.append(acc)

print("LOAO-CV accuracy:", np.mean(cv_accs))

# %%
