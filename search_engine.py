#-------------------------------------------------------------
# AUTHOR: Ivan Trinh
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #1
# TIME SPENT: about like a few hours
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------

with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------

print("Documents:", documents)

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------

stemmer = PorterStemmer()

stop_words = ['i', 'she', 'he', 'they', 'the', 'and', 'his', 'her', 'their', 'and', 'a', 'an', 'this']

def tokenizer(text):
    tokens = text.lower().split()
    filtered_tokens = []
    for token in tokens:
        token = token.strip(".,!?")
        if token not in stop_words:
            stemmed = stemmer.stem(token)
            filtered_tokens.append(stemmed)
    return filtered_tokens
    

vectorizer = CountVectorizer(analyzer = 'word',tokenizer = tokenizer, token_pattern = None, ngram_range = (1, 2), binary = True)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------

vectorizer.fit(documents)
document_matrix = vectorizer.transform(documents)

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------

query = ["I love dogs"]
query_vector = vectorizer.transform(query)

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
# --> add your Python code here

doc_vectors = document_matrix.toarray().tolist()
query_vector = query_vector.toarray()[0]

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------

scores = []
for doc_vector in doc_vectors:
    score = sum(q*d for q, d in zip(query_vector, doc_vector))
    scores.append(int(score))

print("Scores:", scores)

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------

ranking = sorted(enumerate(scores), key = lambda x: x[1], reverse = True)

print("Ranking (doc_index, score): ", ranking)