#!/usr/bin/env python
# coding: utf-8

import re
import math
import numpy as np
import pandas as pd
import nltk
from string import digits
from nltk.stem import PorterStemmer
from os import listdir
from collections import defaultdict

FILE_PATH = "./data/"

STEMMER = PorterStemmer()

r = open("stopwords.txt")
STOPWORDS = r.read()

# Documents preprocessing
def preprocessing(file):
    # Preprocessing
    doc = file.replace("\s+"," ").replace("\n", "").replace("\r\n", "")
    doc = re.sub(r"[^\w\s]", "", doc)
    doc = re.sub(r"[0-9]", "", doc)
    doc = doc.replace("_", "")

    # Tokenization and Lowercasing
    tokenization = [word.lower() for word in doc.split(" ")]

    # Stemming using Porter's Algorithm
    stemming = [STEMMER.stem(word) for word in tokenization]

    # Stopword Removal
    result = [word for word in stemming if word not in STOPWORDS]

    return result


# Get term frequency
def count_tf(doc_set):
    tf_all = list()

    for doc in doc_set:
        token_list = preprocessing(doc)
        
        tf = dict()
        for term in token_list:
            if term in tf:
                tf[term] += 1
            else:
                tf[term] = 1
        tf_all.append(tf)

    return tf_all


# Load documents
files = listdir(FILE_PATH)
files.sort(key=lambda x: int(x[:-4]))
doc_set = list()

# Read files
for file in files:
    with open(FILE_PATH + file, "r") as f:
        document_id = str(file)[:-4]
        document = f.read()
        doc_set.append([document_id, document])
        
# Get training data
# Select those had been classified as training data
labels = dict()
with open("training.txt", "r") as f:
    line = f.readline().strip()
    while line:
        data = line.split(" ")
        label = data[0]
        doc_list = data[1:]
        labels[label] = doc_list
        line = f.readline().strip()

training_set = list()
for label in labels:
    for doc_id in labels[label]:
        training_set.append(doc_set[int(doc_id)-1] + [label])

training_df = pd.DataFrame(training_set)
training_df.columns = ["doc_id", "document", "label"]
training_df = training_df.astype({"doc_id": "int", "label": "int"})
training_df = training_df.sort_values(by="doc_id")
training_df = training_df.reset_index(drop = True)


# Get testing data
test_set = list()
for doc in doc_set:
    doc_id = doc[0]
    if int(doc_id) not in list(training_df["doc_id"]):
        test_set.append(doc_set[int(doc_id)-1] + [None])

test_df = pd.DataFrame(test_set)
test_df.columns = ["doc_id", "document", "label"]
test_df = test_df.astype({"doc_id": "int"})
test_df = test_df.sort_values(by="doc_id")
test_df = test_df.reset_index(drop = True)


# Update training and testing data set
training_df["tf"] = count_tf(training_df["document"])
training_df = training_df[["doc_id", "document", "tf", "label"]]
test_df["tf"] = count_tf(test_df["document"])
test_df = test_df[["doc_id", "document", "tf", "label"]]

def extract_vocabulary(token_lists):
    return {token for token_list in token_lists for token in token_list}

# Chi square test
def chi_square(labels, dataset):
    vocabulary = extract_vocabulary(dataset.tf)
    N = len(dataset)
    chi2 = dict()
    count = 0
    for term in vocabulary:
        chi2_term = 0
        matrix = dict()
        matrix["tp"] = dataset[dataset["tf"].apply(lambda x: term in x)]
        matrix["ta"] = dataset[dataset["tf"].apply(lambda x: term not in x)]
        for lb in labels:
            matrix["cp"] = dataset[dataset["label"] == int(lb)]
            matrix["ca"] = dataset[dataset["label"] != int(lb)]
            matrix["tp_cp"] = len(matrix["tp"][matrix["tp"]["label"] == int(lb)])
            matrix["tp_ca"] = len(matrix["tp"][matrix["tp"]["label"] != int(lb)])
            matrix["ta_cp"] = len(matrix["ta"][matrix["ta"]["label"] == int(lb)])
            matrix["ta_ca"] = len(matrix["ta"][matrix["ta"]["label"] != int(lb)])
            chi2_class = 0
            for i in ["tp", "ta"]:
                for j in ["cp", "ca"]:
                    E = len(matrix[i]) * len(matrix[j]) / N
                    chi2_class += ((matrix[f"{i}_{j}"] - E)**2) / E
            chi2_term += chi2_class
        chi2[term] = chi2_term
        count += 1
        if count % 500 == 0:
            print(f"Finish: {count}/{len(vocabulary)}")
    # Use only 500 terms, chosen by ranking
    vocabulary = sorted(chi2, key=chi2.get, reverse=True)[:500] 
    return vocabulary

vocabulary = extract_vocabulary(training_df.tf)
print(f"Vocabulary size before feature selection: {len(vocabulary)}")

vocabulary = chi_square(labels, training_df)
print(f"Vocabulary size after feature selection: {len(vocabulary)}")

# NB Model
def train_multinominal_nb(labels, dataset, vocabulary):
    n_docs = len(dataset)
    prior = dict()
    cond_prob = {term: dict() for term in vocabulary}
    
    for lb in labels:
        n_class_docs = len(labels[lb])
        class_docs = dataset[dataset["label"] == int(lb)]
        tct = dict()
        # 事前機率
        prior[c] = n_class_docs / n_docs
        # 事後機率
        for term in vocabulary:
            tokens_of_term = 0
            for tf in class_docs["tf"]:
                if term in tf:
                    tokens_of_term += tf[term]
            tct[term] = tokens_of_term
        for term in vocabulary:
            cond_prob[term][lb] = (tct[term]+1) / (sum(tct.values())+len(vocabulary))
            
    return vocabulary, prior, cond_prob

def apply_multinomial_nb(document, labels, vocabulary, prior, cond_prob):
    tf = document["tf"]
    score = dict()
    
    for lb in labels:
        score[lb] = math.log2(prior[lb])
        for term in tf:
            if term in vocabulary:
                score[lb] += (math.log2(cond_prob[term][c]))*(tf[term])
            
    return max(score, key=score.get)


# Train and Test
vocabulary, prior, cond_prob = train_multinominal_nb(labels, training_df, vocabulary)

test_df["label"] = test_df.apply(
    apply_multinomial_nb, C=labels, vocabulary=vocabulary, prior=prior, cond_prob=cond_prob, axis=1)


# Output
output_df = test_df[["doc_id", "label"]]
output_df.columns = ["Id", "Value"]
output_df.to_csv("output.csv", index=False)

