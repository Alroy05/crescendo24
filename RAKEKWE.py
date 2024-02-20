from rake_nltk import Rake
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')

import pandas as pd

csv_file_path = 'toothpaste-final.csv'
df = pd.read_csv(csv_file_path)
column_name = 'review_body'
reviews =' '
reviews = ' '.join(df[column_name].astype(str))

r = Rake()

r.extract_keywords_from_text(reviews)
# print(r)

# print('Ranked list:- \n') 
rankedList = r.get_ranked_phrases_with_scores()
print(rankedList)

print('Extracted Words')
extractedWords = r.get_ranked_phrases()
# print(extractedWords)

phraseString = ','.join(extractedWords)
# print(phraseString)

keywordList =[]

for keyword in rankedList:
  keyword_updated       = keyword[1].split()
  keyword_updated_string    = " ".join(keyword_updated[:2])
  keywordList.append(keyword_updated_string)
  if(len(keywordList)>10):
    break

print(keywordList)