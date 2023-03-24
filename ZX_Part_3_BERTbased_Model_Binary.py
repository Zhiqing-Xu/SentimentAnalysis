#!/usr/bin/env python
# coding: utf-8
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# The following code ensures the code work properly in 
# MS VS, MS VS CODE and jupyter notebook on both Linux and Windows.
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
if __name__ == "__main__":
    print("="*80)
    if os.name == 'nt' or platform == 'win32':
        print("Running on Windows")
        if 'ptvsd' in sys.modules:
            print("Running in Visual Studio")
#--------------------------------------------------#
    if os.name != 'nt' and platform != 'win32':
        print("Not Running on Windows")
#--------------------------------------------------#
    if "__file__" in globals().keys():
        print('CurrentDir: ', os.getcwd())
        try:
            os.chdir(os.path.dirname(__file__))
        except:
            print("Problems with navigating to the file dir.")
        print('CurrentDir: ', os.getcwd())
    else:
        print("Running in python jupyter notebook.")
        try:
            if not 'workbookDir' in globals():
                workbookDir = os.getcwd()
                print('workbookDir: ' + workbookDir)
                os.chdir(workbookDir)
        except:
            print("Problems with navigating to the workbook dir.")
#--------------------------------------------------#

###################################################################################################################
###################################################################################################################
# Imports
#--------------------------------------------------#
import re
import csv
import time
import copy
import torch
import string
import pickle
import random
import pickle
import requests
import argparse
import numpy as np
import pandas as pd

#--------------------------------------------------#
import urllib
import xml.etree.ElementTree as ET
from urllib.request import urlopen
#--------------------------------------------------#
import seaborn as sns
import matplotlib.pyplot as plt 

#--------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
#--------------------------------------------------#
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
#--------------------------------------------------#
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
#--------------------------------------------------#
from collections import defaultdict

#--------------------------------------------------#
# Pretrained Models.
# Topic Models
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings

# Language Models.
import transformers

from transformers import Trainer
from transformers import TrainingArguments

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from transformers import RobertaTokenizerFast
from transformers import RobertaForSequenceClassification

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from transformers import DistilBertTokenizer
from transformers import DistilBertForSequenceClassification

# from ignite.metrics import Accuracy, Precision, Recall, Fbeta

#--------------------------------------------------#
from ZX_Course_Project_utils import clean_text, clean_text_preproc, clean_text_LM, beautiful_print



#--------------------------------------------------#
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Yb. `7MM"""YMM   .M"""bgd   .g8"""bgd `7MM"""Mq. `7MMF'`7MM"""Mq. MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'                    #
#                 MM    `Yb. MM    `7  ,MI    "Y .dP'     `M   MM   `MM.  MM    MM   `MM.P'   MM   `7   MM .dP'    `YM. MMN.    M                      #
#                 MM     `Mb MM   d    `MMb.     dM'       `   MM   ,M9   MM    MM   ,M9      MM        MM dM'      `MM M YMb   M                      #
#                 MM      MM MMmmMM      `YMMNq. MM            MMmmdM9    MM    MMmmdM9       MM        MM MM        MM M  `MN. M                      #
#                 MM     ,MP MM   Y  , .     `MM MM.           MM  YM.    MM    MM            MM        MM MM.      ,MP M   `MM.M                      #
#                 MM    ,dP' MM     ,M Mb     dM `Mb.     ,'   MM   `Mb.  MM    MM            MM        MM `Mb.    ,dP' M     YMM                      #
#               .JMMmmmdP' .JMMmmmmMMM P"Ybmmd"    `"bmmmd'  .JMML. .JMM.JMML..JMML.        .JMML.    .JMML. `"bmmd"' .JML.    YM                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# Part 2 - Extra
'''
BERT-based model trained on Dataset #0 and tested on Dataset #1 and Dataset #2.
'''






#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    `7MM"""Mq.`7MM"""Mq. `7MM"""YMM  `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd      #
#      MM   `MM. MM   `MM.  MM    `7    MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M      #
#      MM   ,M9  MM   ,M9   MM   d      MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       `      #
#      MMmmdM9   MMmmdM9    MMmmMM      MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM               #
#      MM        MM  YM.    MM   Y  ,   MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF'    #
#      MM        MM   `Mb.  MM     ,M   MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM      #
#    .JMML.    .JMML. .JMM.JMMmmmmMMM .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#====================================================================================================#
def Load_dataset_0():
    # Load dataset #1 with "ISO-8859-1" encoding.
    df_raw_0 = pd.read_csv(filepath_or_buffer   =   data_folder/data_file_0           , 
                           header               =   0                                 , 
                           #index_col           =   0                                 , 
                           encoding             =   "ISO-8859-1"                      , 
                           sep                  =   ','                               , 
                           low_memory           =   False                             , 
                           )


    # Print size of the raw dataset.
    print("\n\n" + "="*80)
    # print("\n" + "=" * 80 + "\nRaw dataset printed below: (Only showing useful columns)")
    print("df_raw_0.shape: ", df_raw_0.shape) # 4 columns. 
    # Remove Useless columnns directly.
    df_raw_0 = df_raw_0.drop(columns=["ID", ]) 
    df_raw_0 = df_raw_0.rename(columns={"text": "comments",})
    return df_raw_0

#====================================================================================================#
def Load_dataset_1():
    # Load dataset #1 with "ISO-8859-1" encoding.
    df_raw_1 = pd.read_csv(filepath_or_buffer   =   data_folder/data_file_1           , 
                           header               =   0                                 , 
                           index_col            =   0                                 , 
                           encoding             =   "ISO-8859-1"                      , 
                           sep                  =   ','                               , 
                           low_memory           =   False                             , 
                           )


    # Print size of the raw dataset.
    print("\n\n" + "="*80)
    # print("\n" + "=" * 80 + "\nRaw dataset printed below: (Only showing useful columns)")
    print("df_raw_1.shape: ", df_raw_1.shape) # 4 columns. 
    # Remove Useless columnns directly.
    df_raw_1 = df_raw_1.drop(columns=["date", "post_id", "comment_id"]) 
    return df_raw_1

#====================================================================================================#
def Load_dataset_2():
    # Load dataset #2 with "ISO-8859-1" encoding.
    df_raw_2 = pd.read_csv(filepath_or_buffer   =   data_folder/data_file_2           ,
                           header               =   0                                 , 
                           #index_col           =   0                                 ,
                           encoding             =   "ISO-8859-1"                      ,
                           sep                  =   ','                               ,  
                           low_memory           =   False                             , 
                           )

    # Print dimension of the raw data.
    # print("\n\n" + "=" * 80 + "\nRaw dataset printed below: (Only showing useful columns)")
    print("df_raw_2.shape: ", df_raw_2.shape) # 8 columns. 
    # Remove Useless columnns directly.
    df_raw_2 = df_raw_2.drop(columns=[ "score" , "id" , "url" , "comms_num" , "created" , "timestamp" ]) 
    df_raw_2.loc[df_raw_2["title"] == "Comment", "title"] = ""
    # Combine title and comments since title contains useful info.
    df_raw_2['comments'] = df_raw_2["title"].astype(str) + " . " + df_raw_2["body"].astype(str)
    def fix_string(s):
        return s[3:] if s.startswith(" . ") else s
    df_raw_2['comments'] = df_raw_2['comments'].apply(fix_string)
    return df_raw_2

#====================================================================================================#
def Load_dataset_3():
    # Load dataset #2 with "ISO-8859-1" encoding.
    df_raw_3 = pd.read_csv(filepath_or_buffer   =   data_folder/data_file_3           ,
                           header               =   0                                 , 
                           index_col            =   0                                 ,
                           encoding             =   "ISO-8859-1"                      ,
                           sep                  =   ','                               ,  
                           low_memory           =   False                             , 
                           )

    # Print dimension of the raw data.
    # print("\n\n" + "=" * 80 + "\nRaw dataset printed below: (Only showing useful columns)")
    print("df_raw_3.shape: ", df_raw_3.shape) # 8 columns. 
    # Remove Useless columnns directly.
    df_raw_3 = df_raw_3.drop(columns=[ "ID" ]) 
    df_raw_3 = df_raw_3.rename(columns={"Tweet": "comments",})
    # Combine title and comments since title contains useful info.

    return df_raw_3






#====================================================================================================#
def Clean_data(df_raw_n, num_words_lb = 2, remove_meaningless = True): # num_words_lb is the minimum number of words in one sentence.

    # Get a copy of the raw.
    df_cleaned_n = copy.deepcopy(df_raw_n) 

    # Drop nan values in the columns.
    df_cleaned_n = df_cleaned_n.dropna() 

    # Rename columns.
    df_cleaned_n['cleaned_text'] = df_cleaned_n['comments'] 
    df_cleaned_n = df_cleaned_n.drop(columns=["comments"])  

    # Make a copy.
    df_cleaned_n_all = df_cleaned_n.copy(deep = True)

    # Clean.
    df_cleaned_n_all['cleaned_text'] = df_cleaned_n_all['cleaned_text']     \
                                                .apply(clean_text_preproc)  \
                                                .apply(clean_text_LM)       \
                                                .apply(clean_text_preproc)

    # Check duplicates
    print("\n\n" + "=" * 125 + "\nCheck for duplicates (mostly deleted/short answer): ")
    print(df_cleaned_n_all['cleaned_text'].value_counts().reset_index()) # Lots of duplciates

    # Identify language. (Skipped)
    #   - Use a saved processed dataframe if found in the savings, otherwise generate a new one. 
    #   - Skipped since non-English text are negligible.

    # DROP DUPLICATES !
    df_cleaned_n_all = df_cleaned_n_all.drop_duplicates(subset = ['cleaned_text'], keep = 'first')


    # Check Meaningless Comments
    if remove_meaningless == True:
        text_list_n = df_cleaned_n_all['cleaned_text'].values.tolist()
        for text_x in text_list_n:
            if len(text_x) == 1 or text_x == "" or text_x == "thank you" or text_x.count(" ") == num_words_lb-1:
                # Remove one-word answer. May contain sentiment but require context to understand exact meaning.
                # Even two-words can be meaningful. For example, fuck p***n.
                text_list_n.remove(text_x)
            if text_x.find("i am a bot") != -1: # Remove bot messages.
                text_list_n.remove(text_x)
            
            # Removing all short answer can be a option.
            """
            if text_x.count(" ") <= 2:
                text_list_1.remove(text_x)
            """
        df_cleaned_n_all = pd.DataFrame(text_list_n, columns = ['cleaned_text'])

    print("\n\n" + "=" * 125 + "\nCleaned Dataset: ")
    beautiful_print(df_cleaned_n_all)

    return df_cleaned_n_all

#====================================================================================================#


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#  `7MM"""Yp, `7MM"""YMM  `7MM"""Mq. MMP""MM""YMM *MM                                       `7MM      `7MMM.     ,MMF'              `7MM         `7MM  #
#    MM    Yb   MM    `7    MM   `MM.P'   MM   `7  MM                                         MM        MMMb    dPMM                  MM           MM  #
#    MM    dP   MM   d      MM   ,M9      MM       MM,dMMb.   ,6"Yb.  ,pP"Ybd  .gP"Ya    ,M""bMM        M YM   ,M MM  ,pW"Wq.    ,M""bMM  .gP"Ya   MM  #
#    MM"""bg.   MMmmMM      MMmmdM9       MM       MM    `Mb 8)   MM  8I   `" ,M'   Yb ,AP    MM        M  Mb  M' MM 6W'   `Wb ,AP    MM ,M'   Yb  MM  #
#    MM    `Y   MM   Y  ,   MM  YM.       MM mmmmm MM     M8  ,pm9MM  `YMMMa. 8M"""""" 8MI    MM        M  YM.P'  MM 8M     M8 8MI    MM 8M""""""  MM  #
#    MM    ,9   MM     ,M   MM   `Mb.     MM       MM.   ,M9 8M   MM  L.   I8 YM.    , `Mb    MM        M  `YM'   MM YA.   ,A9 `Mb    MM YM.    ,  MM  #
#  .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.  .JMML.     P^YbmdP'  `Moo9^Yo.M9mmmP'  `Mbmmd'  `Wbmd"MML.    .JML. `'  .JMML.`Ybmd9'   `Wbmd"MML.`Mbmmd'.JMML.#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


# 


# Create the function to preprocess every tweet
def process_review(review):
    """
    Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet
        """
    # remove old style retweet text "RT"
    review = re.sub(r'^RT[\s]+', '', review)
    # remove hyperlinks
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    review = re.sub(r'#', '', review)
    # removing hyphens
    review = re.sub('-', ' ', review)
    # remove linebreaks
    review = re.sub('<br\s?\/>|<br>', "", review)
    # remving numbers
    review = re.sub(r"(\b|\s+\-?|^\-?)(\d+|\d*\.\d+)\b",'',review)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=True, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(review)
    # remove numbers
    tweet_tokens = [i for i in tweet_tokens if not i.isdigit()]
    tweets_clean = []
    for word in tweet_tokens:
        tweets_clean.append(word)
    return ' '.join(tweets_clean)




class LoadDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.reviews)

    # __getitem__ helps us to get a review out of all reviews
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(review,
                                              add_special_tokens=True,
                                              max_length=self.max_len,
                                              truncation = True,
                                              return_token_type_ids=False,
                                              padding='max_length',
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                              )

        return {
                'review_text': review,
                'input_ids': encoding['input_ids'].flatten(),         # flatten() flattens a continguous range of dims in a tensor
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.long)
                }




















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#   `7MMM.     ,MMF'      db      `7MMF'`7MN.   `7MF'                 M             M             M                                                    #
#     MMMb    dPMM       ;MM:       MM    MMN.    M                   M             M             M                                                    #
#     M YM   ,M MM      ,V^MM.      MM    M YMb   M                   M             M             M                                                    #
#     M  Mb  M' MM     ,M  `MM      MM    M  `MN. M               `7M'M`MF'     `7M'M`MF'     `7M'M`MF'                                                #
#     M  YM.P'  MM     AbmmmqMA     MM    M   `MM.M                 VAMAV         VAMAV         VAMAV                                                  #
#     M  `YM'   MM    A'     VML    MM    M     YMM                  VVV           VVV           VVV                                                   #
#   .JML. `'  .JMML..AMA.   .AMMA..JMML..JML.    YM                   V             V             V                                                    #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
# 
if __name__ == "__main__":

    #====================================================================================================#
    # Args
    Step_code     = "P03A_"

    data_folder   = Path("./data_folder")
    data_file_0   = "sentiment_analysis.csv"
    data_file_1   = "Part_2_Dataset_1_reddit_raw_ukraine_russia.csv"
    data_file_2   = "Part_2_Dataset_2_russian_invasion_of_ukraine.csv"
    data_file_3   = "Part_3_semeval_train.csv"


    saved_preproc_dataset_0 = "Saving_preproc_dataset_0.p"
    saved_preproc_dataset_1 = "Saving_preproc_dataset_1.p"
    saved_preproc_dataset_2 = "Saving_preproc_dataset_2.p"
    reprocess_dataset       = False

    #====================================================================================================#
    # Process Data 

    if os.path.exists(saved_preproc_dataset_0) and not reprocess_dataset: 
        df_cleaned_0_all = pd.read_pickle(saved_preproc_dataset_0)
        print("\n\n" + "=" * 125 + "\nCleaned Part 1 Dataset #0 : ")
        beautiful_print(df_cleaned_0_all)
    else:
        df_raw_0 = Load_dataset_0()
        df_cleaned_0_all = Clean_data(df_raw_0, num_words_lb = 2, remove_meaningless = False)
        df_cleaned_0_all.to_pickle(saved_preproc_dataset_0)


    if os.path.exists(saved_preproc_dataset_1) and not reprocess_dataset: 
        df_cleaned_1_all = pd.read_pickle(saved_preproc_dataset_1)
        print("\n\n" + "=" * 125 + "\nCleaned Part 2 Dataset #1 : ")
        beautiful_print(df_cleaned_1_all)
    else:
        df_raw_1 = Load_dataset_1()
        df_cleaned_1_all = Clean_data(df_raw_1, num_words_lb = 2)
        df_cleaned_1_all.to_pickle(saved_preproc_dataset_1)


    if os.path.exists(saved_preproc_dataset_2) and not reprocess_dataset: 
        df_cleaned_2_all = pd.read_pickle(saved_preproc_dataset_2)
        print("\n\n" + "=" * 125 + "\nCleaned Part 2 Dataset #2 : ")
        beautiful_print(df_cleaned_2_all)
    else:
        df_raw_2 = Load_dataset_2()
        df_cleaned_2_all = Clean_data(df_raw_2, num_words_lb = 2)
        df_cleaned_2_all.to_pickle(saved_preproc_dataset_2)


    # df_cleaned_1_all.to_csv(path_or_buf = "Saving_preproc_dataset_1.csv")
    # df_cleaned_2_all.to_csv(path_or_buf = "Saving_preproc_dataset_2.csv")





    #====================================================================================================#
    # Process Data 

    df = df_cleaned_0_all.rename(columns={"cleaned_text": "review", "label" : "sentiment"})
    df['review_processed'] = df['review'].apply(process_review)

    #
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    # Lets load pre-trained Distill BertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    # Lets use below text to understand tokenization process
    # First I am processing our review using above defined function
    sample_text = process_review(df['review_processed'][0])

    # Lets apply our BertTokenizer on sample text
    tokens = tokenizer.tokenize(sample_text)    # this will convert sentence to list of words
    token_ids = tokenizer.convert_tokens_to_ids(tokens) # this will convert list of words to list of numbers based on tokenizer

    print(f'Sentence   : {sample_text}')
    print(f'Tokens     : {tokens}     ')
    print(f'Token IDs  : {token_ids}  ')

    encoding = tokenizer.encode_plus(
        sample_text,
        max_length=32,  # Here for experiment I gave 32 as max_length
        truncation = True,  # Truncate to a maximum length specified with argument max_length
        add_special_tokens=True, # Add '[CLS]', [PAD] and '[SEP]'
        return_token_type_ids=False,  # since our use case deals with only one sentence as opposed to use case which use 2 sentences in single training example(for ex: Question-anwering) we can have it as false
        padding='max_length',   # pad to longest sequence as defined by max_length
        return_attention_mask=True,  # Returns attention mask. Attention mask indicated to the model which tokens should be attended to, and which should not.
        return_tensors='pt',  # Return PyTorch tensors
        )

    print(len(encoding['input_ids'][0]))
    encoding['input_ids'][0]

    # Attention mask also has same length. Zero's in output if any says those corresponds to padding
    print(len(encoding['attention_mask'][0]))
    encoding['attention_mask']

    max_len = 512


    # Lets have 70% for training, 15% for validation and 15% for testing
    X_train, X_valid, y_train, y_valid = \
        train_test_split(df[['review_processed','review']]      ,
                            df['sentiment']                     ,
                            stratify     = df['sentiment']      ,
                            test_size    = 0.30                 ,
                            random_state = 0
                            )

    df_train = pd.concat([pd.DataFrame({'review': X_train['review_processed'].values,'review_old':X_train['review'].values}),pd.DataFrame({'sentiment': y_train.values})], axis = 1)

    df_valid = pd.concat([pd.DataFrame({'review': X_valid['review_processed'].values,'review_old':X_valid['review'].values}),pd.DataFrame({'sentiment': y_valid.values})], axis = 1)



    X_valid, X_test, y_valid, y_test = \
        train_test_split(df_valid[['review','review_old']]         ,
                            df_valid['sentiment']                  ,
                            stratify     = df_valid['sentiment']   ,
                            test_size    = 0.5                     ,
                            random_state = 0
                            )

    df_valid = pd.concat([pd.DataFrame({'review': X_valid['review'].values,'review_old':X_valid['review_old'].values}),pd.DataFrame({'sentiment': y_valid.values})], axis = 1)

    df_test = pd.concat([pd.DataFrame({'review': X_test['review'].values,'review_old':X_test['review_old'].values}),pd.DataFrame({'sentiment': y_test.values})], axis = 1)


    print(df_train.shape, df_valid.shape, df_test.shape)





    def create_data_loader(df, tokenizer, max_len, batch_size):
        ## pass in entire data set here
        ds = LoadDataset(reviews=df.review.to_numpy(),
                         targets=df.sentiment.to_numpy(),
                         tokenizer=tokenizer,
                         max_len=max_len
                        )
        
        # this returns dataloaders with what ever batch size we want
        return DataLoader(ds,
                          batch_size=batch_size,
                          num_workers=4 )                 
        # tells data loader how many sub-processes to use for data loading. 
        # No hard and fast rule. Have to experiment on how many num_workers giving better speed up
                        

    batch_size = 40      # Bert recommendation

    train_data_loader = create_data_loader(df_train, tokenizer, max_len, batch_size)
    valid_data_loader = create_data_loader(df_valid, tokenizer, max_len, batch_size)
    test_data_loader  = create_data_loader(df_test , tokenizer, max_len, batch_size)



    # data = next(iter(train_data_loader))

    # print(data['input_ids'].shape)
    # print(data['attention_mask'].shape)
    # print(data['input_ids'].shape)


    # Lets build classifier for our reviews now. Below n_classes would be 2 in our case since we are classifying review as either positive or negative.

    model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2)
    model = model.to(device)


    EPOCHS = 10

    optimizer = AdamW(model.parameters(), lr = 5e-5)
    total_steps = len(train_data_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(optimizer                         ,
                                                num_warmup_steps = 0              ,
                                                num_training_steps = total_steps  , )


    # Lets write a function to train our model on one epoch
    def train_epoch(model, data_loader, optimizer, device, scheduler, n_examples):

        model = model.train()    # tells your model that we are training
        losses = []
        correct_predictions = 0
        len_d_l = len(data_loader)
        for idx, d in enumerate(data_loader):
            print(f"{idx} / {len_d_l}")
            input_ids      = d["input_ids"]      .to(device)
            attention_mask = d["attention_mask"] .to(device)
            targets        = d["targets"]        .to(device)

            loss, logits = model(input_ids      = input_ids      ,
                                 attention_mask = attention_mask ,
                                 labels         = targets        ,
                                 return_dict    = False
                                 )
        
            # logits : classification scores befroe softmax
            # loss   : classification loss
            
            #print(loss, logits)

            logits = logits.cpu().detach().numpy()
            label_ids = targets.to('cpu').numpy()

            preds = np.argmax(logits, axis=1).flatten()   #returns indices of maximum logit
            targ = label_ids.flatten()

            correct_predictions += np.sum(preds == targ)

            losses.append(loss.item())
            loss.backward()        # performs backpropagation(computes derivates of loss w.r.t to parameters)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  #clipping gradients so they dont explode
            optimizer.step()       #After gradients are computed by loss.backward() this makes the optimizer iterate over all parameters it is supposed to update and use internally #stored grad to update their values
            scheduler.step()       # this will make sure learning rate changes. If we dont provide this learning rate stays at initial value
            optimizer.zero_grad()  # clears old gradients from last step

        return correct_predictions / n_examples, np.mean(losses)



    # Lets write a function to validate our model on one epoch
    def eval_model(model, data_loader, device, n_examples):
    
        model = model.eval()   # tells model we are in validation mode
        losses = []
        correct_predictions = 0

        with torch.no_grad():
            for d in data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                targets = d["targets"].to(device)
                loss, logits = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels = targets,
                                    return_dict    = False
                                    )


                logits = logits.detach().cpu().numpy()
                label_ids = targets.to('cpu').numpy()

                preds = np.argmax(logits, axis=1).flatten()
                targ = label_ids.flatten()

                correct_predictions += np.sum(preds == targ)
                losses.append(loss.item())

        return correct_predictions / n_examples, np.mean(losses)



    # standard block
    # used accuracy as metric here
    history = defaultdict(list)

    best_acc = 0

    for epoch in range(EPOCHS):

        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, optimizer, device, scheduler, len(df_train))

        print(f'Train loss {train_loss} Accuracy {train_acc}')

        val_acc, val_loss = eval_model(model, valid_data_loader, device, len(df_valid))

        print(f'Val   loss {val_loss} Accuracy {val_acc}')
        print()

        history['train_acc' ].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'   ].append(val_acc)
        history['val_loss'  ].append(val_loss)

        if val_acc > best_acc:
            torch.save(model.state_dict(), 'Saving_distilBert_transferlearning')
            best_acc = val_acc
        # We are storing state of best model indicated by highest validation accuracy



    # lets load trained model

    path1 = "./Saving_distilBert_transferlearning"
    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    model = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2)
    model.load_state_dict(torch.load(path1))

    model = model.to(device)    # moving model to device. Device here can be GPU or CPU depending on availability

    test_acc, _ = eval_model(model, test_data_loader, device,len(df_test))
    test_acc.item()



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#




#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#   `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'       `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'      `7M'M`MF'  #
#     VAMAV          VAMAV          VAMAV          VAMAV          VAMAV           VAMAV          VAMAV          VAMAV          VAMAV          VAMAV    #
#      VVV            VVV            VVV            VVV            VVV             VVV            VVV            VVV            VVV            VVV     #
#       V              V              V              V              V               V              V              V              V              V      #

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
###################################################################################################################
###################################################################################################################
#====================================================================================================#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#--------------------------------------------------#
#------------------------------

#                                                                                                                                                          
#      `MM.              `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.             `MM.       
#        `Mb.              `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.             `Mb.     
# MMMMMMMMMMMMD     MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD    MMMMMMMMMMMMD   
#         ,M'               ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'              ,M'     
#       .M'               .M'              .M'              .M'              .M'              .M'              .M'              .M'              .M'       
#                                                                                                                                                          

#------------------------------
#--------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#====================================================================================================#
###################################################################################################################
###################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#       A              A              A              A              A               A              A              A              A              A      #
#      MMM            MMM            MMM            MMM            MMM             MMM            MMM            MMM            MMM            MMM     #
#     MMMMM          MMMMM          MMMMM          MMMMM          MMMMM           MMMMM          MMMMM          MMMMM          MMMMM          MMMMM    #
#   ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.       ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.      ,MA:M:AM.  #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
#       M              M              M              M              M               M              M              M              M              M      #
