import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords

# nltk.download('stopwords')

def extract_keywords(csv_file_path):
    df = pd.read_csv(csv_file_path)
    column_name = 'review_body'
    documents = df[column_name].astype(str).tolist()

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    names = vectorizer.get_feature_names_out()

    data = vectors.todense().tolist()
    df = pd.DataFrame(data, columns=names)

    st = set(stopwords.words('english'))
    df = df.loc[:, ~df.columns.isin(st)]

    selected_words = []

    for _, row in df.iterrows():
        selected_words += row[row > 0.5].index.tolist()

    # Remove duplicates and convert to a comma-separated string
    selected_words_string = ', '.join(list(dict.fromkeys(selected_words)))

    return selected_words_string

st.title('Keyword Extraction with TF-IDF')

# List of CSV files
csv_files = ['listerine.csv', 'mouthwash.csv', 'toothbrush.csv', 'toothpaste.csv']

# Dropdown menu
csv_file_path = st.selectbox('Select a CSV file:', csv_files)

if st.button('Extract Keywords'):
    selected_words_string = extract_keywords(csv_file_path)
    st.write('Selected Words:', selected_words_string)
