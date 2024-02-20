import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

csv_file_path = 'toothpaste-final.csv'
df = pd.read_csv(csv_file_path)
column_name = 'review_body'

# Combine all text from the 'review_body' column into a list
documents = df[column_name].astype(str).tolist()

# Create TfidfVectorizer and fit_transform on the list of documents
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(documents)
names = vectorizer.get_feature_names_out()

# Convert the vectors to a dense matrix and create a DataFrame
data = vectors.todense().tolist()
df = pd.DataFrame(data, columns=names)

nltk.download('stopwords')
st = set(stopwords.words('english'))

# Remove columns containing stop words
df = df.loc[:, ~df.columns.isin(st)]

N = 10

# Initialize an empty string to store selected words
selected_words_string = ""

for _, row in df.iterrows():
    # Filter words with TF-IDF > 0.5
    selected_words = row[row > 0.5].index.tolist()
    
    # Concatenate selected words into a string
    selected_words_string += ' '.join(selected_words) + ' '

print(selected_words_string)
