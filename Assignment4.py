import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pymongo
from sklearn.metrics.pairwise import cosine_similarity

# MongoDB connection
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["search_engine"]
doc_collection = db["documents"]
index_collection = db["inverted_index"]

# Sample documents (based on the example you provided)
documents = [
    "After the medication, headache and nausea were reported by the patient.",
    "The medication caused a headache and nausea, but no dizziness was reported.",
    "Headache and dizziness are common effects of this medication.",
    "The medication caused a headache and nausea, but no dizziness was reported."
]

# Function to tokenize and remove punctuation
def tokenize(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = text.split()  # Split into words
    bigrams = [' '.join([words[i], words[i+1]]) for i in range(len(words)-1)]
    trigrams = [' '.join([words[i], words[i+1], words[i+2]]) for i in range(len(words)-2)]
    return words + bigrams + trigrams  # Return unigrams, bigrams, and trigrams

# Step 2: Build Inverted Index
def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: defaultdict(list))  # {term: {doc_id: [positions]}}
    term_document_frequency = defaultdict(int)
    tfidf_vectorizer = TfidfVectorizer()

    # Tokenize and build inverted index
    for doc_id, doc in enumerate(documents):
        tokens = tokenize(doc)
        for pos, token in enumerate(tokens):
            inverted_index[token][doc_id].append(pos)
            term_document_frequency[token] += 1
    
    return inverted_index, term_document_frequency

# Build the inverted index and document frequency
inverted_index, term_document_frequency = build_inverted_index(documents)

# Insert documents into MongoDB collection
for doc_id, doc in enumerate(documents):
    doc_collection.insert_one({
        "_id": doc_id,
        "content": doc
    })

# Insert inverted index into MongoDB collection
for term, doc_dict in inverted_index.items():
    for doc_id, positions in doc_dict.items():
        term_data = {
            "_id": term,
            "pos": doc_id,  # Document ID
            "docs": [{"doc_id": doc_id, "tfidf": 0.5, "positions": positions}]  # Placeholder for TF-IDF value
        }
        index_collection.insert_one(term_data)

# Compute TF-IDF for documents
def compute_tfidf_for_documents(documents):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    return tfidf_matrix, tfidf_vectorizer

# Query ranking function
def rank_documents(query, inverted_index, tfidf_matrix, tfidf_vectorizer, doc_collection):
    query_vector = tfidf_vectorizer.transform([query])
    scores = []

    # Retrieve matching terms from the inverted index
    terms_in_query = tokenize(query)

    # Loop over the terms in the query and find matching documents
    for term in terms_in_query:
        if term in inverted_index:
            for doc in inverted_index[term]:
                doc_data = doc_collection.find_one({"_id": doc})
                tfidf_values = [doc_data["tfidf"]]  # Placeholder logic for now, adjust based on TF-IDF calculation
                cosine_score = cosine_similarity(query_vector, tfidf_matrix[doc])
                scores.append((doc_data["content"], cosine_score))
    
    # Sort the documents by cosine similarity score
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores

# Example Queries
queries = [
    "nausea and dizziness",
    "effects",
    "nausea was reported",
    "dizziness",
    "the medication"
]

# Get the TF-IDF matrix for documents
tfidf_matrix, tfidf_vectorizer = compute_tfidf_for_documents(documents)

# Rank documents for each query
for query in queries:
    ranked_docs = rank_documents(query, inverted_index, tfidf_matrix, tfidf_vectorizer, doc_collection)
    for doc, score in ranked_docs:
        print(f"{doc}, {score}")
