import pickle
from xml.etree import ElementTree
import fnmatch
import shutil
import argparse
import base64
from datetime import datetime
import hashlib
import logging
import os
import pickle
import re
import sys
import time
import numpy as np
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import RegexpTokenizer
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def load_pan_data(xmls_directory, truth_path):
    xml_filenames = os.listdir(xmls_directory)
    #print(xml_filenames[0])
    truths, author_ids = load_truth(truth_path)
    #print(len(author_ids))
    files=[]
    for i in author_ids:
        files.append(str(i)+".xml")
    # print(len(truths))
    # print(author_ids[0])
    # print(truths[0])
    # print(files[0])
    original_tweet_lengths = []  
    merged_tweets_of_authors = [] 
    for author_index, xml_filename in enumerate(files):
        if fnmatch.fnmatch(xml_filename, '*.xml'):
            tree = ElementTree.parse(os.path.join(xmls_directory, xml_filename),
                                     parser=ElementTree.XMLParser(encoding="utf-8"))
            root = tree.getroot()
            original_tweet_lengths.append([])
            tweets_of_this_author = []  
            for child in root[0]:
                tweet = child.text
                original_tweet_lengths[author_index].append(len(tweet))
                tweet = tweet.replace('\n', " <LineFeed> ")
                tweets_of_this_author.append(tweet)
            merged_tweets_of_this_author = " <EndOfTweet> ".join(tweets_of_this_author) + " <EndOfTweet>"
            merged_tweets_of_authors.append(preprocess_tweet(merged_tweets_of_this_author))
    return merged_tweets_of_authors, truths, author_ids, original_tweet_lengths

def load_pan_data_tira(xmls_directory):
    xml_filenames = sorted(os.listdir(xmls_directory))
    author_ids = []
    for xml_filename in xml_filenames:
        author_ids.append(xml_filename[:-4])
    
    original_tweet_lengths = []  
    merged_tweets_of_authors = []
    for author_index, xml_filename in enumerate(xml_filenames):
        if fnmatch.fnmatch(xml_filename, '*.xml'):
            tree = ElementTree.parse(os.path.join(xmls_directory, xml_filename),
                                     parser=ElementTree.XMLParser(encoding="utf-8"))
            root = tree.getroot()
            original_tweet_lengths.append([])
            tweets_of_this_author = []  
            for child in root[0]:
                tweet = child.text
                original_tweet_lengths[author_index].append(len(tweet))
                tweet = tweet.replace('\n', " <LineFeed> ")
                tweets_of_this_author.append(tweet)
            merged_tweets_of_this_author = " <EndOfTweet> ".join(tweets_of_this_author) + " <EndOfTweet>"
            merged_tweets_of_authors.append(preprocess_tweet(merged_tweets_of_this_author))
    return merged_tweets_of_authors, author_ids, original_tweet_lengths

def load_truth(truth_path):
    sorted_author_ids_and_truths = [] 
    with open(truth_path, 'r') as truth_file:
        for line in sorted(truth_file):
            line = line.rstrip('\n')
            sorted_author_ids_and_truths.append(line.split(":::"))
    truths = []  
    author_ids=[]
    for i, row in enumerate(sorted_author_ids_and_truths):
        # if row[0] == author_ids[i]:
        truths.append(row[2])
        author_ids.append(row[0])
        # else:
            #continue
    return truths , author_ids

def write_predictions_to_xmls(author_ids_test, y_predicted, xmls_destination_main_directory, language_code):
    xmls_destination_directory = os.path.join(xmls_destination_main_directory, language_code)
    os.makedirs(xmls_destination_directory, exist_ok=True)
    for author_id, predicted_gender in zip(author_ids_test, y_predicted):
        if (predicted_gender=='bot'):
            Type='bot'
        else:
            Type='Human'
        root = ElementTree.Element('author', attrib={'id': author_id,
                                                     'lang': language_code,
                                                     'type':Type,
                                                     'gender': predicted_gender
                                                     })
        tree = ElementTree.ElementTree(root)
        tree.write(os.path.join(xmls_destination_directory, author_id + ".xml"))

def load_datasets(input_directory, preset_key):
    PRESETS_DICTIONARY = {'PAN19_English': {'dataset_name': 'PAN 2019 English',
                                            'xmls_subdirectory': 'en/',
                                            'train_subpath': 'en/truth-train.txt',
                                            'test_subpath': 'en/truth-dev.txt',
                                            'xmls_test': 'en/',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]
    TRAINING_DATASET_MAIN_DIRECTORY =input_directory
    xmls_directory = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['xmls_subdirectory'])
    truth_path_train = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['train_subpath'])
    truth_path_test = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['test_subpath'])

    docs_train, y_train, author_ids_train, original_tweet_lengths_train = \
        load_pan_data(xmls_directory, truth_path_train)
    docs_test, y_test, author_ids_test, original_tweet_lengths_test = \
        load_pan_data(xmls_directory, truth_path_test)
    #print(len(docs_train))
    #print(len(docs_test))

    return docs_train, docs_test, y_train, author_ids_test

def load_datasets_tira(input_directory,preset_key):
    PRESETS_DICTIONARY = {'PAN19_English': {'dataset_name': 'PAN 2019 English',
                                            'xmls_test': 'en/',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]
    TRAINING_DATASET_MAIN_DIRECTORY =input_directory
    xmls_directory = os.path.join(TRAINING_DATASET_MAIN_DIRECTORY, PRESET['xmls_test'])
    docs_test, author_ids_test, original_tweet_lengths_test = \
        load_pan_data_tira(xmls_directory)
    return docs_test, author_ids_test

FLAGS = re.MULTILINE | re.DOTALL
def preprocess_tweet(text):
    eyes = r"[8:=;]"
    nose = r"['`\-]?"
    def re_sub(pattern, repl):
        return re.sub(pattern, repl, text, flags=FLAGS)
    text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " ")
    text = re_sub(r"(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))", " em_positive ") 
    text = re_sub(r"(:\s?D|:-D|x-?D|X-?D)", " em_positive ") 
    text = re_sub(r"(<3|:\*)", " em_positive ") 
    text = re_sub(r"(;-?\)|;-?D|\(-?;)", " em_positive ") 
    text = re_sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', " em_negative ")
    text = re_sub(r'(:,\(|:\'\(|:"\()', " em_negative ")
    text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " em_positive ")
    text = re_sub(r"{}{}p+".format(eyes, nose), " em_positive ")
    text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " em_negative ")
    text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " em_neutralface ")
    text = re_sub(r" 👎 "," em_negative ")
    text = re_sub(r"ha?ha", r" em_positive ")

    tweet=text

    word_tokens = word_tokenize(tweet)  
    filtered_tweet = [] 

    for w in word_tokens: 
        if w not in stop_words: 
            filtered_tweet.append(w)
    tweet=("".join(filtered_tweet)).lower()
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
    tokens = tokenizer.tokenize(tweet)
    detokenizer = TreebankWordDetokenizer()
    processed_tweet = detokenizer.detokenize(tokens)
    return(processed_tweet)

def extract_features(docs_train, docs_test, preset_key):
    PRESETS_DICTIONARY = {'PAN19_English': {'dataset_name': 'PAN 2019 English',
                                            'word_ngram_range': (1, 3),
                                            'perform_dimentionality_reduction': True,
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]
    word_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                      analyzer='word', ngram_range=PRESET['word_ngram_range'],
                                      min_df=2, use_idf=True, sublinear_tf=True)
    char_vectorizer = TfidfVectorizer(preprocessor=preprocess_tweet,
                                     analyzer='char', ngram_range=(3, 5),
                                     min_df=2, use_idf=True, sublinear_tf=True)

    if docs_train is not None:
        X_train_ngrams_tfidf = word_vectorizer.fit_transform(docs_train)
        X_test_ngrams_tfidf = word_vectorizer.transform(docs_test)
    else:
        X_test_ngrams_tfidf = word_vectorizer.fit_transform(docs_test)
    if PRESET['perform_dimentionality_reduction']:
        svd = TruncatedSVD(n_components=300, random_state=42)
        if docs_train is not None:
            X_train_ngrams_tfidf_reduced = svd.fit_transform(X_train_ngrams_tfidf)
            X_test_ngrams_tfidf_reduced = svd.transform(X_test_ngrams_tfidf)
        else:
            X_test_ngrams_tfidf_reduced = svd.fit_transform(X_test_ngrams_tfidf)
        if docs_train is not None:
            X_train = X_train_ngrams_tfidf_reduced
        X_test = X_test_ngrams_tfidf_reduced
    else:
        if docs_train is not None:
            X_train = X_train_ngrams_tfidf
        X_test = X_test_ngrams_tfid
    if docs_train is not None:
        return X_train,X_test
    else:
        return X_test

def cross_validate_model(clf, X_train, y_train):
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, scoring='accuracy', cv=skf)
    print("scores = ",scores)
    print("Accuracy = ",scores.mean()*100)

def train_model_and_predict(clf, X_train, y_train, X_test, author_ids_test, preset_key,
                            write_to_xml_files=True, xmls_destination_directory=None, ):
    PRESETS_DICTIONARY = {'PAN19_English': {'dataset_name': 'PAN 2019 English',
                                            'language_code': 'en',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]

    #print("X-train:\n",X_train)
    clf.fit(X_train, y_train)
    f="pickle.pkl"
    pickle.dump(clf,open(f,'wb'))
    y_predicted = clf.predict(X_test)
    #print("Y-predicted:\n",y_predicted)
    if write_to_xml_files:
        write_predictions_to_xmls(author_ids_test, y_predicted,xmls_destination_directory, PRESET['language_code'])
        
def predict_tira(X_test, author_ids_test, preset_key, write_to_xml_files=True, xmls_destination_directory=None):
    PRESETS_DICTIONARY = {'PAN19_English': {'dataset_name': 'PAN 2019 English',
                                            'language_code': 'en',
                                            },
                          }
    PRESET = PRESETS_DICTIONARY[preset_key]
    pkl_filename="pickle.pkl"
    with open(pkl_filename, 'rb') as file:  
        pickle_model = pickle.load(file)
    clf=pickle_model
    y_predicted = clf.predict(X_test)
    if write_to_xml_files:
        write_predictions_to_xmls(author_ids_test, y_predicted,xmls_destination_directory, PRESET['language_code'])

def main_evaluation():
    command_line_argument_parser = argparse.ArgumentParser()
    command_line_argument_parser.add_argument("-i")
    command_line_argument_parser.add_argument("-o")
    command_line_arguments = command_line_argument_parser.parse_args()
    input_directory = command_line_arguments.i
    output_directory = command_line_arguments.o
    preset_key="PAN19_English"
    docs_train, docs_test, y_train, author_ids_test =load_datasets(input_directory, preset_key)
    docs_train = docs_train[:1000]
    docs_test = docs_test[:200]
    y_train = y_train[:1000]
    author_ids_test = author_ids_test[:200]
    X_train,X_test = extract_features(docs_train,docs_test, preset_key)
    clf = LinearSVC(random_state=42)
    #print("lens:",len(y_train),len(X_train))


    cross_validate_model(clf, X_train, y_train)
    train_model_and_predict(clf, X_train, y_train, X_test, author_ids_test, preset_key,False, output_directory)
    print("Total Time %.2f seconds", time.process_time())
    
def TIRA_evaluation():
    command_line_argument_parser = argparse.ArgumentParser()
    command_line_argument_parser.add_argument("-c")
    command_line_argument_parser.add_argument("-o")
    command_line_arguments = command_line_argument_parser.parse_args()
    test_dataset_main_directory = command_line_arguments.c
    prediction_xmls_destination_main_directory = command_line_arguments.o
    preset_key="PAN19_English"
    docs_test,author_ids_test =load_datasets_tira(test_dataset_main_directory, preset_key)
    X_test = extract_features(None,docs_test, preset_key)
    
    predict_tira(X_test, author_ids_test, preset_key, True, prediction_xmls_destination_main_directory)

if __name__ == "__main__":
    #main_evaluation()
    TIRA_evaluation()
