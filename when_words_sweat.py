import pandas as pd
import os, json
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
loanstats = pd.read_csv("LoanStats.csv", encoding="ISO-8859-1")



loanstats['FullyPaid'] = np.where(loanstats['Status'] == "Fully Paid", 1, 0)

X_train, X_test, y_train, y_test = train_test_split(
    loanstats['Loan Description'], loanstats['FullyPaid'], test_size=0.1, random_state=42)


train_df_text = X_train

#Really long array
train_df_text1 = [i for i in train_df_text]

#train_df_text1

#bag of words, no context words
from gensim.parsing.preprocessing import preprocess_string
toy_corpus = [preprocess_string(str(doc)) for doc in train_df_text1]

#Dictionary can encode the words to numbers
from gensim import corpora
dictionary = corpora.Dictionary(toy_corpus)
#print(dictionary.token2id)

from gensim import models
import time

#(Word id, frequency)
toy_corpus = [dictionary.doc2bow(text) for text in toy_corpus]
toy_corpus


#Some random model
start_time4 = time.time()
lda8 = models.LdaModel(toy_corpus, 
                      id2word=dictionary,
                      iterations=1000,
                      num_topics=8)
lda8.print_topics()    
elapsed_time4 = time.time() - start_time4


#Weights
lda_corpus = lda8[toy_corpus]

#Same but for test

from gensim.parsing.preprocessing import preprocess_string
test_df_text = X_test

test_df_text1 = [i for i in test_df_text]

toy_corpus_test = [preprocess_string(str(doc)) for doc in test_df_text1]

toy_corpus_test = [dictionary.doc2bow(text) for text in toy_corpus_test]

lda_corpus_test = lda8[toy_corpus_test]


# A matrix of features with i being the document and j being the topic, there are 8
lda_features_train = np.zeros((len(lda_corpus), 8))
for i, row in enumerate(lda_corpus):
    for j, value in row:
        lda_features_train[i, j] = value

#Same

lda_features_test = np.zeros((len(lda_corpus_test), 8))
for i, row in enumerate(lda_corpus_test):
    for j, value in row:
        lda_features_test[i, j] = value

# for i, j in enumerate(lda_corpus):
#     print(i, j)

# Random Forest model should be run on lda_features_train, which is now suitable for fitting
# y_train is just 0 or 1
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42)
rfc.fit(lda_features_train, y_train)
y_pred = rfc.predict(lda_features_test)

#Predict on lda_features_test which went through the same preprocessing steps ast lda_features_train

print("Accuracy is", accuracy_score(y_test, y_pred))
print("Precision is", precision_score(y_test, y_pred))
print("Recall is", recall_score(y_test, y_pred))





