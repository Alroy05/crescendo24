import streamlit as st
from rake_nltk import Rake
import nltk
import pandas as pd

# nltk.download('stopwords')
# nltk.download('punkt')

def extract_keywords(csv_file_path):
    df = pd.read_csv(csv_file_path)
    column_name = 'review_body'
    reviews =' '
    reviews = ' '.join(df[column_name].astype(str))

    r = Rake()
    r.extract_keywords_from_text(reviews)

    rankedList = r.get_ranked_phrases_with_scores()

    extractedWords = r.get_ranked_phrases()

    phraseString = ','.join(extractedWords)

    keywordList =[]

    for keyword in rankedList:
        keyword_updated       = keyword[1].split()
        keyword_updated_string    = " ".join(keyword_updated[:2])
        keywordList.append(keyword_updated_string)
        if(len(keywordList)>10):
            break

    return keywordList

st.title('Keyword Extraction with Rake')

# List of CSV files
csv_files = ['listerine.csv', 'mouthwash.csv', 'toothbrush.csv', 'toothpaste.csv']

# Dropdown menu
csv_file_path = st.selectbox('Select a CSV file:', csv_files)

if st.button('Extract Keywords'):
    keywordList = extract_keywords(csv_file_path)
    st.write('Top 10 Keywords:', ', '.join(keywordList))
