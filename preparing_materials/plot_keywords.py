
# %%
import collections
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import HDBSCAN
# %%
terms = [
    "conversation analysis",
    "language policy",
    "language ideology and attitudes",
    "Decision making",
    "AI",
    "Cognitive Development",
    "Gender",
    "Childhood",
    "Discourse analysis",
    "Research in neurodiversity",
    "Social dynamics",
    "Decision making",
    "sentiment analysis",
    "cognition",
    "psycholinguistics",
    "Philosophy of mind",
    "Hypnosis",
    "Inferential Statistics",
    "Pragmatic Analysis",
    "Natural Language Processing",
    "Sentiment analysis",
    "literature",
    "cognition",
    "neuroscience",
    "sentiment analysis",
    "computational political research",
    "semantic extraction",
    "Psycholinguistics",
    "Digital intimacy",
    "Organizational culture",
    "semiotics",
    "philosophy of deconstruction",
    "doublespeak",
    "Social media research",
    "Sock puppet auditing",
    "NLP that supports social media research"
]

# just added these fields as visual anchors
fields = [
    "cognitive science",
    "linguistics",
    "literary studies",
    "religion",
    "philosophy",
    "computer science",
    #"neuroscience",
    "political science",
    "sociology",
    "media studies",
    "nlp",
    "psychology",
    "creative writing",
    #"critical theory",
    "computational linguistics",
    "data science"
]

# normalize
terms = [t.lower().strip() for t in terms]

# term frequencies (for point size later)
freq = collections.Counter(terms)

# now get unique terms
terms = list(set(terms))

# --- embeddings ---
model = SentenceTransformer("all-MiniLM-L6-v2")
#embeddings = model.encode(terms)

all_terms = terms + fields
embeddings = model.encode(all_terms)


# %%
# dim red
reducer = umap.UMAP(
    n_neighbors=6,
    min_dist=0.3,
    metric="cosine",
    random_state=42)

# get coords
coords = reducer.fit_transform(embeddings)
student_coords = coords[:len(terms)]
anchor_coords = coords[len(terms):]

# clustering? optional, not implem
clusterer = HDBSCAN(min_cluster_size=3, metric="euclidean")
clusters = clusterer.fit_predict(coords)

# %%
plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

# scale point sizes by freq
student_sizes = [80 + freq[t] * 100 for t in terms] # we want the ones that were chosen more often to be larger
anchor_sizes = [300 for _ in fields] # just sizes for the anchor words

plt.scatter(student_coords[:, 0], student_coords[:, 1],
            s=student_sizes, alpha=0.6)

plt.scatter(anchor_coords[:, 0], anchor_coords[:, 1],
            s=anchor_sizes, marker="X", alpha=0.9)


# label anchors
for i, field in enumerate(fields):
    plt.text(anchor_coords[i, 0], anchor_coords[i, 1], field, fontsize=11, fontweight="bold")

for i, term in enumerate(terms):
    #if freq[term] > 1 or len(term) > 25:
    plt.text(student_coords[i, 0], student_coords[i, 1], term, fontsize=9, alpha=0.85)

plt.title("Student Interests (Embedding-based)")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.show()


# %%
