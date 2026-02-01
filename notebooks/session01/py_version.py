# %%

import requests

# %%
# SETTINGS
output_file = "results2.txt"

ebook_number = "345"

# %%
# we go get our requests object from gutendex:)
r = requests.get(f"https://gutendex.com/books/{ebook_number}")

# we convert this to a dictionary by using json
data = r.json()

# as you know dictionary structure helps us because we can get a picture of what data we have
print(data.keys())

# %%

text_url = data['formats']['text/plain; charset=us-ascii']
text = requests.get(text_url).text
# %%

# awesome, now we have the text and we have the metadata
print(data['title'])
print(data['authors'])
print(data['languages'])
print(data['download_count'])

print("\n")
print(text[:700])

# %%

# let's write some of our metadata into a txt file in your working directory
with open(output_file, 'w') as f:
    f.write(f"Title: {data['title']}\n")
    f.write(f"Authors: {(data['authors'])}\n")
    f.write(f"Download count: {data['download_count']}\n")

# %%