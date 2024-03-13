import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv('processed_data/processed_data_gemini-pro_emo_valence.csv')

# Display the first few rows of the dataframe
# print(df.head())


nltk.download('punkt')
nltk.download('stopwords')

# Tokenization and preprocessing
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Tokenize and convert to lowercase
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens

df['processed_text'] = df['gpt_response'].apply(preprocess_text)

# Combine all tokens into a single list
all_tokens = sum(df['processed_text'].tolist(), [])

# Calculate frequencies
word_freq = Counter(all_tokens)

# Display the 10 most common words
print(word_freq.most_common(10))

# Get the 20 most common words and their counts
most_common_words = word_freq.most_common(20)
words = [word[0] for word in most_common_words]
counts = [word[1] for word in most_common_words]

# Plotting
plt.figure(figsize=(10, 8))
plt.bar(words, counts)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.title('Top 20 Most Common Words')
plt.show()