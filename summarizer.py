import torch
import re
import nltk
import heapq
import spacy
import pandas as pd
from collections import Counter
from transformers import pipeline


def transformer_summarizer(text):
    summarizer = pipeline("summarization")
    res = summarizer(text)
    return res[0]['summary_text']

def nltk_summarizer(article_text, nsentences=7):
    # Removing Square Brackets and Extra Spaces
    article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
    article_text = re.sub(r'\s+', ' ', article_text)
    # # Removing special characters and digits
    formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
    formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)
    # formatted_article_text = article_text
    sentence_list = nltk.sent_tokenize(article_text)
    stopwords = nltk.corpus.stopwords.words('english')
    temp = nltk.word_tokenize(formatted_article_text)
    word_frequencies = {}
    for word in temp:
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

        print('Scores', sentence_scores)

    summary_sentences = heapq.nlargest(nsentences, sentence_scores, key=sentence_scores.get)
    print('HERE', summary_sentences)
    summary = ' '.join(summary_sentences)
    print('Here', summary)
    return summary

# ------- Spacy ---------

# Define function to cleanup text by removing personal pronouns, stopwords, and puncuation



# df_questions['Body_Cleaned'] = df_questions['Body_Cleaned_1'].apply(lambda x: cleanup_text(x, False))

def generate_summary(text_without_removing_dot, nsentences=7):
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    stopwords = nltk.corpus.stopwords.words('english')
    nlp = spacy.load('en_core_web_lg')
    def cleanup_text(docs, logging=False):
        texts = []
        doc = nlp(docs, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
        return pd.Series(texts)
    
    cleaned_text = cleanup_text(text_without_removing_dot)[0]
    # print('CLEAN', cleaned_text)
    sample_text = text_without_removing_dot
    doc = nlp(sample_text)
    sentence_list=[]
    for idx, sentence in enumerate(doc.sents): # we are using spacy for sentence tokenization
        print(sentence)
        sentence_list.append(re.sub(r'[^\w\s]','',str(sentence)))

    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}  
    for word in nltk.word_tokenize(cleaned_text):  
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1


    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():  
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)


    sentence_scores = {}  
    # print(word_frequencies)
    for sent in sentence_list:  
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                # if len(sent.split(' ')) < 30:
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word]
                else:
                    sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(nsentences, sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    # print("Original Text::::::::::::\n")
    # print(len(text_without_removing_dot))
    # print(text_without_removing_dot)
    # print('\n\nSummarized text::::::::\n')
    # print(len(summary))
    # print(summary)
    return summary

