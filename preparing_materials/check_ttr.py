# %%
import pandas as pd

# %%

# %%
df = pd.read_excel("/Users/au324704/Library/Mobile Documents/com~apple~CloudDocs/2023_CHCAA/FABULANET/Resources/CHICAGO_MEASURES_MARCH24.xlsx")
df.head()
# %%
# remove nans in wordcounts
df = df[df['WORDCOUNT'].notna()]

# top rating_counts:
col = "MSTTR-100"
df.sort_values(col, ascending=False)[['TITLE', 'MSTTR-100', 'WORDCOUNT']].tail(60)
# %%

kerouac = df[df['AUTH_LAST'] == 'Kerouac']
kerouac[['TITLE', 'MSTTR-100', 'WORDCOUNT']]
# %%
