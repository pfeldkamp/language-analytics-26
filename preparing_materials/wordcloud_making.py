# %%
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
# %%

# read text file
with open("interest_list.txt", "r", encoding="utf-8") as f:
    keywords = [line.strip() for line in f if line.strip()]

keywords
# %%

# count duplicates
freqs = Counter(keywords)

# generate word cloud from frequencies
wc = WordCloud(
    width=800,
    height=400,
    background_color="white",
    colormap="tab20",
    collocations=False
).generate_from_frequencies(freqs)

# display
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
# %%
