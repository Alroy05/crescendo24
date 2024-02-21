import streamlit as st
from rake_nltk import Rake
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# nltk.download('stopwords')
# nltk.download('punkt')

def extract_keywords_rake(csv_file_path):
    df = pd.read_csv(csv_file_path)
    column_name = 'review_body'
    reviews =' '
    reviews = ' '.join(df[column_name].astype(str))

    r = Rake()
    r.extract_keywords_from_text(reviews)

    rankedList = r.get_ranked_phrases_with_scores()

    keywordList =[]

    for keyword in rankedList:
        keyword_updated       = keyword[1].split()
        keyword_updated_string    = " ".join(keyword_updated[:2])
        keywordList.append(keyword_updated_string)
        if(len(keywordList)>10):
            break

    # Convert the list of keywords to a DataFrame
    df_keywords = pd.DataFrame(keywordList, columns=['Keyword'])

    return df_keywords

def extract_keywords_tfidf(csv_file_path):
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

    # Remove duplicates
    selected_words = list(dict.fromkeys(selected_words))

    # Convert the list of selected words to a DataFrame
    df_selected_words = pd.DataFrame(selected_words, columns=['Selected Word'])

    return df_selected_words

st.title('Keyword Extraction')

# List of CSV files
csv_files = ['listerine.csv', 'mouthwash.csv', 'toothbrush.csv', 'toothpaste.csv']

# Dropdown menu
csv_file_path = st.sidebar.selectbox('Select a CSV file:', csv_files)

# Model selection
model = st.sidebar.selectbox('Select a model:', ['Rake', 'TF-IDF'])

if st.sidebar.button('Extract Keywords'):
    if model == 'Rake':
        df_keywords = extract_keywords_rake(csv_file_path)
        st.dataframe(df_keywords)  # Display the DataFrame in a tabular form
        
        # Create a word cloud
        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white', 
                        stopwords = stopwords.words('english'), 
                        min_font_size = 10).generate(' '.join(df_keywords['Keyword']))
        st.image(wordcloud.to_array(), use_column_width=True)
        
        # Pie chart for keyword distribution
        keyword_counts = Counter(df_keywords['Keyword'])
        fig, ax = plt.subplots()
        ax.pie(keyword_counts.values(), labels=keyword_counts.keys(), autopct='%1.1f%%')
        ax.set_title('Keyword Distribution')
        st.pyplot(fig)
        
        # Frequency distribution plot
        fig, ax = plt.subplots()
        ax.bar(keyword_counts.keys(), keyword_counts.values())
        ax.set_title('Frequency Distribution of Keywords')
        ax.set_xlabel('Keyword')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)

    else:
        df_selected_words = extract_keywords_tfidf(csv_file_path)
        st.dataframe(df_selected_words)  # Display the DataFrame in a tabular form
        
        # Create a word cloud
        wordcloud = WordCloud(width = 800, height = 800, 
                        background_color ='white', 
                        stopwords = stopwords.words('english'), 
                        min_font_size = 10).generate(' '.join(df_selected_words['Selected Word']))
        st.image(wordcloud.to_array(), use_column_width=True)
        
        # Pie chart for selected word distribution
        selected_word_counts = Counter(df_selected_words['Selected Word'])
        fig, ax = plt.subplots()
        ax.pie(selected_word_counts.values(), labels=selected_word_counts.keys(), autopct='%1.1f%%')
        ax.set_title('Selected Word Distribution')
        st.pyplot(fig)
        
        # Frequency distribution plot
        fig, ax = plt.subplots()
        ax.bar(selected_word_counts.keys(), selected_word_counts.values())
        ax.set_title('Frequency Distribution of Selected Words')
        ax.set_xlabel('Selected Word')
        ax.set_ylabel('Frequency')
        ax.tick_params(axis='x', rotation=90)
        st.pyplot(fig)
