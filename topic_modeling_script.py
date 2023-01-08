import pandas as pd

# import data set
quora = pd.read_csv('quora_questions.csv')
quora.head()

# npl tf-idf vectorazation
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = tfidf.fit_transform(quora['Question'])
dtm

# non-negative matrix factorization
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=20,random_state=42)
nmf_model.fit(dtm)
topic_results = nmf_model.transform(dtm)

# Display topic modeling results
topic_results.argmax(axis=1)
quora['Topic'] = topic_results.argmax(axis=1)
quora.head(10)