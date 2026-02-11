# %%
# Import packages
from glob import glob
import re
import nltk
import requests
import matplotlib.pyplot as plt

# %%
# PART 1
# Print 'Hellow world'
print("Hello world!")
# %%
# Define first-line regex pattern
first_line_pattern = re.compile(
    r"^[^.]*\.", re.IGNORECASE
)  # Matches all characters until the first . (included)

# Open Frankenstien
with open("../../../DATA/gb/pg84.txt") as f:
    frankenstein_text = f.read()

# Extract matches using defiend regex patterns
match = first_line_pattern.search(frankenstein_text)

# If the pattern has extracted matches from the text, keep only the first match
if match:
    first_line = match.group(0)
else:
    first_line = None  # If no match is found, set the variable to 'None'

# Print the contenct of the variable
print(first_line)
# %%
# Loop through all files in the folder an exectute the same pattern matching x printing operation
for filename in glob("../../../DATA/gb/*.txt"):
    with open(filename) as f:
        raw_text = f.read()

    match = first_line_pattern.search(raw_text)
    first_line = (
        match.group(0) if match else None
    )  # This logic can also be compressed into a single line like this

    print(first_line)
# %%
# PART 2A - ADVANCED
# Split book into chapters
# Compile a splitter that splits the book into chapters
chapter_splitter = re.compile(r"\nCHAPTER \d+", re.IGNORECASE)

# Split chapters
chapters = chapter_splitter.split(frankenstein_text)
# %%
# Remove everything (preface and letters) until the first chapter starts
clean_frankenstein_text = re.sub(
    r"^.*?\nCHAPTER 1", "", frankenstein_text, flags=re.IGNORECASE | re.DOTALL
)  # Note: The re.DOTALL flag ensures that the . (all characters) pattern also matches newlines, \n (which are usually exempt)

# %%
# Tokenise that text and count words in each chapter
# Download tokenisation model
nltk.download("punkt")
nltk.download("punkt_tab")
# %%
# Import word_tokenze function
from nltk.tokenize import word_tokenize

# Tokenise text into seperate words
tokens = word_tokenize(clean_frankenstein_text)
# %%
# Count work counts for each chapter
# Loop through
for chapter in chapters:
    # Tokenize the chapter
    tokens = word_tokenize(chapter)
    # Count and print word count
    print(len(tokens))
# %%
# PART 2B

# Make lists to store the data
titles = []
download_counts = []
birth_years = []

# Loop through first 100 titles:
for ebook_number in range(1, 31):
    # Open a request to Gutendex
    r = requests.get(f"https://gutendex.com/books/{ebook_number}")

    # Only continue if request was successful (not all books exist in a json format)
    if r.status_code == 200:
        # Convert data to JSON file
        data = r.json()

        # Append metadata
        titles.append(data["title"])
        download_counts.append(data["download_count"])
        authors_list = data.get("authors", [])
        birth_year = authors_list[0].get("birth_year") if authors_list else None
        birth_years.append(birth_year)

# Convert lists with data into combined dictionar
metadata_dict = {
    "title": titles,
    "download_counts": download_counts,
    "birth_years": birth_years,
}
# %%
# Print the dictionary
print(metadata_dict)

# %%
# Plotting the relationships between author birth year and download counts
birth_years = metadata_dict["birth_years"]
download_counts = metadata_dict["download_counts"]

# Remove missing birth years
filtered_birth_years = []
filtered_download_counts = []

for year, count in zip(birth_years, download_counts):
    if year is not None:
        filtered_birth_years.append(year)
        filtered_download_counts.append(count)

# %%
# Create a scatterplot using matplotlib
plt.figure()
plt.scatter(filtered_birth_years, filtered_download_counts)

plt.xlabel("Author Birth Year")
plt.ylabel("Download Count")
plt.title("Birth Year vs Download Count")

plt.show()
# %%
# Plot relation between word count and download count
# Make lists to store the data
titles = []
download_counts = []
word_counts = []

# Loop through first 100 titles:
for ebook_number in range(1, 31):
    # Open a request to Gutendex
    r = requests.get(f"https://gutendex.com/books/{ebook_number}")

    # Only continue if request was successful (not all books exist in a json format)
    if r.status_code == 200:
        # Get eraw text
        formats = data.get("formats", {})
        text_url = None

        for key, value in formats.items():
            if key.startswith("text/plain"):
                text_url = value
                break

        # Skip book if no plain text format exists
        if not text_url:
            continue

        raw_text = requests.get(text_url).text

        # Append metadata (only after we know text exists)
        data = r.json()
        titles.append(data["title"])
        download_counts.append(data["download_count"])

        # Tokenise
        tokens = word_tokenize(raw_text)

        # Count words and append to list
        word_counts.append(len(tokens))

# Convert lists with data into combined dictionar
word_counts_dict = {
    "title": titles,
    "download_counts": download_counts,
    "word_counts": word_counts,
}
# %%
print(word_counts_dict)

# %%
# Plotting the relationships between author birth year and download counts
word_counts = word_counts_dict["word_counts"]
download_counts = word_counts_dict["download_counts"]

# Remove missing birth years
filtered_word_counts = []
filtered_download_counts = []

for word_count, download_count in zip(birth_years, download_counts):
    if word_count is not None:
        filtered_word_counts.append(word_count)
        filtered_download_counts.append(download_count)

# %%
# Create a scatterplot using matplotlib
plt.figure()
plt.scatter(filtered_word_counts, filtered_download_counts)

plt.xlabel("Word Count")
plt.ylabel("Download Count")
plt.title("Word Count vs Download Count")

plt.show()
# %%
