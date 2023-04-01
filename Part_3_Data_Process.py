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
# Vader
import nltk
nltk.downloader.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#--------------------------------------------------#
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

from typing import Optional, Union, Tuple, Type, Sequence, List, Set, Any, TextIO, IO
#--------------------------------------------------#
from Part_3_utils import *

import warnings
warnings.filterwarnings("ignore")

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Yb. `7MM"""YMM   .M"""bgd   .g8"""bgd `7MM"""Mq. `7MMF'`7MM"""Mq. MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'                    #
#                 MM    `Yb. MM    `7  ,MI    "Y .dP'     `M   MM   `MM.  MM    MM   `MM.P'   MM   `7   MM .dP'    `YM. MMN.    M                      #
#                 MM     `Mb MM   d    `MMb.     dM'       `   MM   ,M9   MM    MM   ,M9      MM        MM dM'      `MM M YMb   M                      #
#                 MM      MM MMmmMM      `YMMNq. MM            MMmmdM9    MM    MMmmdM9       MM        MM MM        MM M  `MN. M                      #
#                 MM     ,MP MM   Y  , .     `MM MM.           MM  YM.    MM    MM            MM        MM MM.      ,MP M   `MM.M                      #
#                 MM    ,dP' MM     ,M Mb     dM `Mb.     ,'   MM   `Mb.  MM    MM            MM        MM `Mb.    ,dP' M     YMM                      #
#               .JMMmmmdP' .JMMmmmmMMM P"Ybmmd"    `"bmmmd'  .JMML. .JMM.JMML..JMML.        .JMML.    .JMML. `"bmmd"' .JML.    YM                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


"""
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                                                                                                                                                      #
#    > Part 3 - Instruction (Copied from Cource Project Instruction)                                                                                   #
#                                                                                                                                                      #
#    Based on your sentiment analysis results,                                                                                                         #
#        (especially looking at posts/tweets with negative and positive sentiment,                                                                     #
#        as posts/tweets with neutral sentiment are less informative),                                                                                 #
#    you will need to IDENTIFY factors/reasons/TOPICS that drive sentiment.                                                                            #
#                                                                                                                                                      #
#    Those are factors (reasons, TOPICS) that explain sentiment and can be used                                                                        #
#    for decision making and recommendations in Part 4.                                                                                                #
#                                                                                                                                                      #
#    You can use any Natural Language Processing models for this part of the project                                                                   #
#    (use existing models or develop your own).                                                                                                        #
#                                                                                                                                                      #
#                                                                                                                                                      #
#    > What Does This File Do? (Must Read !!!)                                                                                                         #
#                                                                                                                                                      #
#    (1) Clean the datasets to be analyzed in Part 3.                                                                                                  #
#    (2) Use different sentiment analysis models to analyze comments in the dataset and generate labels of sentiments and emotions.                    #
#    (3) Use a topic analysis model to generate a list of topics for each dataset and classify each comment into a topic.                              #
#                                                                                                                                                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
"""


data_folder = Path("./data_folder")
data_file_1   = "Part_2_Dataset_1_reddit_raw_ukraine_russia.csv"
data_file_2   = "Part_2_Dataset_2_russian_invasion_of_ukraine.csv"


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
def Load_dataset_1():
    """
    Load dataset #1 with "ISO-8859-1" encoding.
    """
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
    """
    Load dataset #2 with "ISO-8859-1" encoding.
    """
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
def Clean_data(df_raw_n, num_words_lb): # num_words_lb is the minimum number of words in one sentence.
    """
    This Function cleans the dataset(s) to be analyzed in Part 3.
    The data processing steps are pretty much the same as Part 1 & 2, including,
    (1) Remove tags.
    (2) Remove non-English comments.
    (3) Remove meaningless comments.
    (4) Remove duplicates.
    """


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
    print("\n\n" + "=" * 125 + "\nCleaned Part 2 Dataset: ")
    beautiful_print(df_cleaned_n_all)

    return df_cleaned_n_all

#====================================================================================================#














#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#                     .M"""bgd `7MM"""YMM  `7MN.   `7MF'MMP""MM""YMM `7MMF'`7MMM.     ,MMF'`7MM"""YMM  `7MN.   `7MF'MMP""MM""YMM                       #
#                    ,MI    "Y   MM    `7    MMN.    M  P'   MM   `7   MM    MMMb    dPMM    MM    `7    MMN.    M  P'   MM   `7                       #
#                    `MMb.       MM   d      M YMb   M       MM        MM    M YM   ,M MM    MM   d      M YMb   M       MM                            #
#                      `YMMNq.   MMmmMM      M  `MN. M       MM        MM    M  Mb  M' MM    MMmmMM      M  `MN. M       MM                            #
#                    .     `MM   MM   Y  ,   M   `MM.M       MM        MM    M  YM.P'  MM    MM   Y  ,   M   `MM.M       MM                            #
#                    Mb     dM   MM     ,M   M     YMM       MM        MM    M  `YM'   MM    MM     ,M   M     YMM       MM                            #
#                    P"Ybmmd"  .JMMmmmmMMM .JML.    YM     .JMML.    .JMML..JML. `'  .JMML..JMMmmmmMMM .JML.    YM     .JMML.                          #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


"""
-- Sentiment Label Prediction via Pretrained Distilled BERT and Vader --

This section include functions that implement three different deeplearning-based sentiment analysis models as follows,
(1) DistillBERT_sentiment:
    - Based on Distill BERT (obviously).
    - Trained by other researchers.
    - Downloaded model weights from `HugginFace`.
    - Predict "pos", "neg", "neu" labels for each comments.

(2) DistillBERT_TR_sentiment:
    - Based on Distill BERT (obviously).
    - We train this Distill BERT Classifier OURSELVES.
    - The model was trained on the dataset in Part 1.
    - Run ZX_Part_3_BERTbased_Model_Binary.py to train the model and save the weights.
    - Predict binary labels for each comments ("pos", "neg").

(3) Vader_Prediction:
    - Trained by other researchers.
    - Implement using `nltk` python package.
    - Predict "pos", "neg", "neu" labels for each comments.

(4) Plutchik Emotion Wheel
    - Based on Distill BERT (obviously).
    - We train this Distill BERT Classifier OURSELVES.
    - Predict "T", "F" labels for 11 different emotions in the `Plutchik Emotion Wheel`.
    - We use a high-quality dataset from Nvidia's `sentiment discovery` research to train the Distill BERT model.
    - The dataset contains labels of 11 different emotions in the `Plutchik Emotion Wheel`.
    - The performance of Distill BERT on the NVIDIA's dataset outperforms all models that has been tried in their paper.

"""



###################################################################################################################
#          `7MM"""Yb.     db            mm     db `7MM `7MM"""Yp, `7MM"""YMM  `7MM"""Mq. MMP""MM""YMM             #
#            MM    `Yb.                 MM          MM   MM    Yb   MM    `7    MM   `MM.P'   MM   `7             #
#            MM     `Mb `7MM  ,pP"Ybd mmMMmm `7MM   MM   MM    dP   MM   d      MM   ,M9      MM                  #
#            MM      MM   MM  8I   `"   MM     MM   MM   MM"""bg.   MMmmMM      MMmmdM9       MM                  #
#            MM     ,MP   MM  `YMMMa.   MM     MM   MM   MM    `Y   MM   Y  ,   MM  YM.       MM                  #
#            MM    ,dP'   MM  L.   I8   MM     MM   MM   MM    ,9   MM     ,M   MM   `Mb.     MM                  #
#          .JMMmmmdP'   .JMML.M9mmmP'   `Mbmo.JMML.JMML.JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.  .JMML.                #
###################################################################################################################
# Sentiment Label Prediction via Pretrained Distilled BERT
def Get_DistillBERT_sentiment_labels(text_list, batch_size = 30):

    DistillBERTmodel     = AutoModelForSequenceClassification.from_pretrained("Souvikcmsa/SentimentAnalysisDistillBERT", )
    DistillBERTtokenizer = AutoTokenizer.from_pretrained("Souvikcmsa/SentimentAnalysisDistillBERT", )
    DistillBERTmodel.cuda()

    labels_list = []
    scores_list = []
    len_text_list = len(text_list)

    print("Start Predicting with DistillBERTmodel ...")
    for i in range(0, len_text_list, batch_size):
        
        last_slice      =   True if i + batch_size >= len_text_list else False
        text_sublist    =   text_list[i : i + batch_size] if not last_slice else text_list[i : ]
        inputs          =   DistillBERTtokenizer(text_sublist, return_tensors = "pt", padding = True, truncation = True)

        input_ids       =  inputs["input_ids"].cuda()
        attention_mask  =  inputs["attention_mask"].cuda()

        outputs      =   DistillBERTmodel(input_ids, attention_mask)
        _, y_tags    =   torch.max(outputs["logits"], dim = 1)
        logits       =   outputs["logits"].cpu().detach().numpy()

        labels_list.append(y_tags.cpu().detach().numpy())
        scores_list.append(logits)
    labels_list  = np.concatenate(labels_list)
    scores_list  = np.concatenate(scores_list)
    return labels_list, scores_list


###################################################################################################################
#       `7MM            mm         `7MM"""Yp, `7MM"""YMM  `7MM"""Mq. MMP""MM""YMM       MMP""MM""YMM `7MM"""Mq.   #
#         MM            MM           MM    Yb   MM    `7    MM   `MM.P'   MM   `7       P'   MM   `7   MM   `MM.  #
#    ,M""bMM  ,pP"Ybd mmMMmm         MM    dP   MM   d      MM   ,M9      MM                 MM        MM   ,M9   #
#  ,AP    MM  8I   `"   MM           MM"""bg.   MMmmMM      MMmmdM9       MM                 MM        MMmmdM9    #
#  8MI    MM  `YMMMa.   MM           MM    `Y   MM   Y  ,   MM  YM.       MM                 MM        MM  YM.    #
#  `Mb    MM  L.   I8   MM           MM    ,9   MM     ,M   MM   `Mb.     MM                 MM        MM   `Mb.  #
#   `Wbmd"MML.M9mmmP'   `Mbmo mmmm .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.  .JMML.     mmmm    .JMML.    .JMML. .JMM. #
###################################################################################################################
# Load a Trained Model based on the transfer learning (based on distilBert.)
def Get_DistillBERT_TR_sentiment_labels(text_list, batch_size = 30, Saved_Model = "Saving_Model/Saving_distilBert_transferlearning.pt"):

    PRE_TRAINED_MODEL_NAME = 'distilbert-base-uncased'
    DistillBERTmodel_TR           = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2)
    DistillBERTmodel_TR_tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

    DistillBERTmodel_TR.load_state_dict(torch.load(Saved_Model))
    DistillBERTmodel_TR.cuda() 

    labels_list = []
    scores_list = []
    len_text_list = len(text_list)

    print("Start Predicting with DistillBERTmodel ...")
    
    for i in range(0, len_text_list, batch_size):
        
        last_slice      =   True if i + batch_size >= len_text_list else False
        text_sublist    =   text_list[i : i + batch_size] if not last_slice else text_list[i : ]
        inputs          =   DistillBERTmodel_TR_tokenizer(text_sublist, return_tensors = "pt", padding = True, truncation = True)

        input_ids       =  inputs["input_ids"].cuda()
        attention_mask  =  inputs["attention_mask"].cuda()

        outputs         =   DistillBERTmodel_TR(input_ids, attention_mask)
        _, y_tags       =   torch.max(outputs["logits"], dim = 1)
        logits          =   outputs["logits"].cpu().detach().numpy()

        labels_list.append(y_tags.cpu().detach().numpy())
        scores_list.append(logits)

    labels_list  = np.concatenate(labels_list)
    scores_list  = np.concatenate(scores_list)

    return labels_list, scores_list



###################################################################################################################
#                            `7MMF'   `7MF' db     `7MM"""Yb. `7MM"""YMM  `7MM"""Mq.                              #
#                              `MA     ,V  ;MM:      MM    `Yb. MM    `7    MM   `MM.                             #
#                               VM:   ,V  ,V^MM.     MM     `Mb MM   d      MM   ,M9                              #
#                                MM.  M' ,M  `MM     MM      MM MMmmMM      MMmmdM9                               #
#                                `MM A'  AbmmmqMA    MM     ,MP MM   Y  ,   MM  YM.                               #
#                                 :MM;  A'     VML   MM    ,dP' MM     ,M   MM   `Mb.                             #
#                                  VF .AMA.   .AMMA.JMMmmmdP' .JMMmmmmMMM .JMML. .JMM.                            #
###################################################################################################################
# Sentiment Label Prediction via Pretrained Vader.
def Vader_Prediction( text_array, threshold = 0.05, model = SentimentIntensityAnalyzer() ):
    """
    Inputs - threshold:
        - Typical threshold values (used in the literature cited on this page) are:
            * positive sentiment: (compound score >= +0.05)
            * neutral  sentiment: (compound score >  -0.05) and (compound score <  +0.05)
            * negative sentiment: (compound score <= -0.05)
        - Using a standardized threshold for compound score is kept as an option.
    Inputs - model: 
        - no other options. 
    
    Outputs: 
        - list of labels. 
    
    See more details here https://github.com/cjhutto/vaderSentiment#About-the-Scoring.
    """

    print("Start Predicting with Vader_Prediction ...")
    sentiment_label_list = []
    for idx, text_x in enumerate(text_array): # Seems that Vader has no batch processig (in parallel) options.
        
        cmpd_score = model.polarity_scores(text_x)['compound']
        if   cmpd_score >=  threshold:
            sentiment_label_list.append("pos")
        elif cmpd_score < -threshold:
            sentiment_label_list.append("neg")
        else:
            sentiment_label_list.append("neu")
    return sentiment_label_list

###################################################################################################################
#                   `7MM"""YMM  `7MMM.     ,MMF' .g8""8q.  MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'             #
#     __,  __,        MM    `7    MMMb    dPMM .dP'    `YM.P'   MM   `7   MM .dP'    `YM. MMN.    M               #
#    `7MM `7MM        MM   d      M YM   ,M MM dM'      `MM     MM        MM dM'      `MM M YMb   M ,pP"Ybd       #
#      MM   MM        MMmmMM      M  Mb  M' MM MM        MM     MM        MM MM        MM M  `MN. M 8I   `"       #
#      MM   MM        MM   Y  ,   M  YM.P'  MM MM.      ,MP     MM        MM MM.      ,MP M   `MM.M `YMMMa.       #
#      MM   MM        MM     ,M   M  `YM'   MM `Mb.    ,dP'     MM        MM `Mb.    ,dP' M     YMM L.   I8       #
#    .JMML.JMML.    .JMMmmmmMMM .JML. `'  .JMML. `"bmmd"'     .JMML.    .JMML. `"bmmd"' .JML.    YM M9mmmP'       #
###################################################################################################################
def Get_11_Sentiment_labels(text_list, batch_size = 30, Saved_Model_folder = Path("./Saving_Model/"), ):
    
    all_sentiment =  ["anger"        ,
                      "anticipation" ,
                      "disgust"      ,
                      "fear"         ,
                      "joy"          ,
                      "love"         ,
                      "optimism"     ,
                      "pessimism"    ,
                      "sadness"      ,
                      "surprise"     ,
                      "trust"        ,]

    all_labels_dict = dict([])
    for idx, sentiment_x in enumerate(all_sentiment):

        PRE_TRAINED_MODEL_NAME        = 'distilbert-base-uncased'
        DistillBERTmodel_TR           = DistilBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL_NAME, num_labels = 2)
        DistillBERTmodel_TR_tokenizer = DistilBertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

        DistillBERTmodel_TR.load_state_dict(torch.load(Saved_Model_folder / ("sentiment_" + sentiment_x + ".pt")))
        DistillBERTmodel_TR.cuda() 

        labels_list = []
        scores_list = []
        len_text_list = len(text_list)

        print("Start Predicting with DistillBERTmodel ...")
        
        for i in range(0, len_text_list, batch_size):
            
            last_slice      =   True if i + batch_size >= len_text_list else False
            text_sublist    =   text_list[i : i + batch_size] if not last_slice else text_list[i : ]
            inputs          =   DistillBERTmodel_TR_tokenizer(text_sublist, return_tensors = "pt", padding = True, truncation = True)

            input_ids       =  inputs["input_ids"].cuda()
            attention_mask  =  inputs["attention_mask"].cuda()

            outputs         =   DistillBERTmodel_TR(input_ids, attention_mask)
            _, y_tags       =   torch.max(outputs["logits"], dim = 1)
            logits          =   outputs["logits"].cpu().detach().numpy()

            labels_list.append(y_tags.cpu().detach().numpy())
            scores_list.append(logits)

        labels_list  = np.concatenate(labels_list)
        scores_list  = np.concatenate(scores_list)
        all_labels_dict[sentiment_x] = labels_list

    return all_labels_dict


def test_Get_11_Sentiment_labels():
    print("Emotions for the following two sentences: " )
    print("I love AutoTrain. " )
    print("I am happy that you called me, but I don't like the way you talk. ")
    print(Get_11_Sentiment_labels(["I love AutoTrain", "I am happy that you called me, but I don't like the way you talk. " ]))



###################################################################################################################
#    .g8"""bgd `7MM"""YMM MMP""MM""YMM    `7MMF'            db     `7MM"""Yp, `7MM"""YMM  `7MMF'       .M"""bgd   #
#  .dP'     `M   MM    `7 P'   MM   `7      MM             ;MM:      MM    Yb   MM    `7    MM        ,MI    "Y   #
#  dM'       `   MM   d        MM           MM            ,V^MM.     MM    dP   MM   d      MM        `MMb.       #
#  MM            MMmmMM        MM           MM           ,M  `MM     MM"""bg.   MMmmMM      MM          `YMMNq.   #
#  MM.    `7MMF' MM   Y  ,     MM           MM      ,    AbmmmqMA    MM    `Y   MM   Y  ,   MM      , .     `MM   #
#  `Mb.     MM   MM     ,M     MM           MM     ,M   A'     VML   MM    ,9   MM     ,M   MM     ,M Mb     dM   #
#    `"bmmmdPY .JMMmmmmMMM   .JMML.       .JMMmmmmMMM .AMA.   .AMMA.JMMmmmd9  .JMMmmmmMMM .JMMmmmmMMM P"Ybmmd"    #
###################################################################################################################
# Sentiment Label Prediction Savings.
def Get_Sentiment_Label_Prediction(df_cleaned_n_all      = None                                     ,
                                   saving_file           = "Saving_Part_3_Sentiment_Label_Pred_.p"  ,
                                   saving_folder         = Path("./")                               ,
                                   force_DistilBERT      = False                                    ,
                                   force_DistilBERT_TR   = False                                    ,
                                   force_Vader           = False                                    ,
                                   force_11_Sentiment    = False                                    ,
                                   Vader_threshold       = 0.05                                     ,
                                   ):

    if os.path.exists(saving_folder / saving_file):
        df_cleaned_n_analysis = pd.read_pickle(saving_folder / saving_file)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Regenerate predicted labels with DistilBERT.
        if force_DistilBERT:
            dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
            df_cleaned_n_analysis["DistilBertLabel"] = Get_DistillBERT_sentiment_labels(dataset_n_text)[0]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Regenerate predicted labels with Vader.
        if force_Vader:
            df_cleaned_n_analysis["Vader_sentiment"] = \
                Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist() , 
                                 threshold = 0.0                                       )
            
            df_cleaned_n_analysis["Vader_sentiment_neu"] = \
                Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist() , 
                                 threshold = Vader_threshold                           )
            
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Regenerate predicted labels with DistilBertLabel_TransferLearning.
        if force_DistilBERT_TR:
            dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
            df_cleaned_n_analysis["DistilBertLabel_TR"] = Get_DistillBERT_TR_sentiment_labels(dataset_n_text)[0]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Get_11_Sentiment_labels
        if force_11_Sentiment:
            dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
            all_labels_dict = Get_11_Sentiment_labels(dataset_n_text)
            for sent_x in list(all_labels_dict.keys()):
                df_cleaned_n_analysis[sent_x] = all_labels_dict[sent_x]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Adjust the Labels to "pos", "neg", "neu".
        label_list = df_cleaned_n_analysis["DistilBertLabel"].values.tolist()
        df_cleaned_n_analysis["DistilBert_sentiment"] = ["neg" if l==0 else "neu" if l==1 else "pos" for l in label_list]

        label_list = df_cleaned_n_analysis["DistilBertLabel_TR"].values.tolist()
        df_cleaned_n_analysis["DistilBert_TR_sentiment"] = ["neg" if l==0 else "pos" for l in label_list]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Save.
        df_cleaned_n_analysis.to_pickle(saving_folder / saving_file)

    else:
        # Make a copy.
        df_cleaned_n_analysis = copy.deepcopy(df_cleaned_n_all)

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # DistilBert Sentiment Labels. (Trained by others.)
        dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
        df_cleaned_n_analysis["DistilBertLabel"] = Get_DistillBERT_sentiment_labels(dataset_n_text)[0]
        label_list = df_cleaned_n_analysis["DistilBertLabel"].values.tolist()
        df_cleaned_n_analysis["DistilBert_sentiment"] = ["neg" if l==0 else "neu" if l==1 else "pos" for l in label_list]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Vader Sentiment Labels. 
        # Use Vader to predict sentiment labels for this dataset. 
        df_cleaned_n_analysis["Vader_sentiment"] = \
            Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist(), 
                             threshold = 0.00                                     )
        # Use Vader to predict sentiment labels (including "neu") for this dataset. 
        df_cleaned_n_analysis["Vader_sentiment_neu"] = \
            Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist(), 
                             threshold = Vader_threshold                          )
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # DistilBert Sentiment TR Labels. (Trained in this project.)
        dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
        df_cleaned_n_analysis["DistilBertLabel_TR"] = Get_DistillBERT_TR_sentiment_labels(dataset_n_text)[0]

        label_list = df_cleaned_n_analysis["DistilBertLabel_TR"].values.tolist()
        df_cleaned_n_analysis["DistilBert_sentiment_TR"] = ["neg" if l==0 else "pos" for l in label_list]   

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Get_11_Sentiment_labels
        dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
        all_labels_dict = Get_11_Sentiment_labels(dataset_n_text)
        for sent_x in list(all_labels_dict.keys()):
            df_cleaned_n_analysis[sent_x] = all_labels_dict[sent_x]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        # Save.
        df_cleaned_n_analysis.to_pickle(saving_folder / saving_file)

    return df_cleaned_n_analysis



###################################################################################################################
#    .g8"""bgd   .g8""8q. `7MMM.     ,MMF'`7MM"""Mq.   db     `7MM"""Mq. `7MMF' .M"""bgd   .g8""8q. `7MN.   `7MF' #
#  .dP'     `M .dP'    `YM. MMMb    dPMM    MM   `MM. ;MM:      MM   `MM.  MM  ,MI    "Y .dP'    `YM. MMN.    M   #
#  dM'       ` dM'      `MM M YM   ,M MM    MM   ,M9 ,V^MM.     MM   ,M9   MM  `MMb.     dM'      `MM M YMb   M   #
#  MM          MM        MM M  Mb  M' MM    MMmmdM9 ,M  `MM     MMmmdM9    MM    `YMMNq. MM        MM M  `MN. M   #
#  MM.         MM.      ,MP M  YM.P'  MM    MM      AbmmmqMA    MM  YM.    MM  .     `MM MM.      ,MP M   `MM.M   #
#  `Mb.     ,' `Mb.    ,dP' M  `YM'   MM    MM     A'     VML   MM   `Mb.  MM  Mb     dM `Mb.    ,dP' M     YMM   #
#    `"bmmmd'    `"bmmd"' .JML. `'  .JMML..JMML. .AMA.   .AMMA.JMML. .JMM.JMML.P"Ybmmd"    `"bmmd"' .JML.    YM   #
###################################################################################################################
# This is a function that compares predicted labels of DistilBERT, DistilBERT_TransferLearning & Vader. 
# A Heatmap is plotted to visualize the confusion matrix. 
def Comparing_predicted_labels(dataframe_analysis = None                    ,
                               column_1           = "DistilBert_sentiment"  ,
                               column_2           = "Vader_sentiment_neu"   ,
                               label_1            = "DistilBert"            ,
                               label_2            = "Vader"                 ,
                               dataset_index      = "_1"                    ,
                               ):
    #====================================================================================================#
    # Percentage HeatMap
    # Compare the two predictions.

    output_file_header = "Part2"
    plot_name = f"{label_2} Prediction vs. {label_1} Prediction"
    results_sub_folder = Path("./")
    column_1_sentiments_1   = dataframe_analysis[column_1] .values.tolist()
    column_2_sentiments_1   = dataframe_analysis[column_2] .values.tolist()

    num_classes = max(len(set(column_1_sentiments_1)), len(set(column_2_sentiments_1)))

    # print(DBERT_sentiments_1)
    # print(Vader_sentiments_1)

    font = {'family' : "Times New Roman"}
    plt.rc('font', **font)
    #--------------------------------------------------#
    cm = confusion_matrix(column_2_sentiments_1, column_1_sentiments_1)
    cm = cm/len(column_1_sentiments_1) 
    confusion_matrix_df = pd.DataFrame(cm) #.rename(columns=idx2class, index=idx2class)
    #--------------------------------------------------#
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.heatmap(confusion_matrix_df, 
                        annot       =  True             , 
                        fmt         =  ".3f"            , 
                        cmap        =  "magma"          , 
                        vmin        =  0.00             , 
                        vmax        =  0.20             , 
                        center      =  0.10             , 
                        cbar_kws    =  {"shrink": .82}  , 
                        linewidths  =  0.1              , 
                        linecolor   =  'gray'           , 
                        annot_kws   =  {"fontsize": 30} , 
                        )

    ax.set_xlabel(f'{label_1} Predicted Class', fontsize = 32)
    ax.set_ylabel(f'{label_2} Predicted Class', fontsize = 32)
    ax.set_title("Confusion Matrix - " \
                + f" \n ({label_2} Prediction vs. {label_1} Prediction)", fontsize = 32)
    if num_classes == 2 :
        ax.xaxis.set_ticklabels([["Negative", "Positive"][i] for i in range(num_classes)], fontsize = 32) 
        ax.yaxis.set_ticklabels([["Negative", "Positive"][i] for i in range(num_classes)], fontsize = 32) 
    elif num_classes == 3 :
        ax.xaxis.set_ticklabels([["Negative", "Neutral", "Positive"][i] for i in range(num_classes)], fontsize = 32) 
        ax.yaxis.set_ticklabels([["Negative", "Neutral", "Positive"][i] for i in range(num_classes)], fontsize = 32)      
    else:
        raise Exception("Number of classes doesnt meet requirement. ")
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.show()
    #--------------------------------------------------#
    saving_name = output_file_header + "_HeatMapCM_" + plot_name + "_percentage"+ dataset_index + ".png"
    saving_name = saving_name 
    fig.savefig(results_sub_folder / saving_name , dpi = 1000 )
    mpl.rcParams.update(mpl.rcParamsDefault)

    return 


















#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#      MMP""MM""YMM   .g8""8q. `7MM"""Mq.`7MMF' .g8"""bgd          db     `7MN.   `7MF'     db     `7MMF'   `YMM'   `MM'.M"""bgd `7MMF' .M"""bgd       #
#      P'   MM   `7 .dP'    `YM. MM   `MM. MM .dP'     `M         ;MM:      MMN.    M      ;MM:      MM       VMA   ,V ,MI    "Y   MM  ,MI    "Y       #
#           MM      dM'      `MM MM   ,M9  MM dM'       `        ,V^MM.     M YMb   M     ,V^MM.     MM        VMA ,V  `MMb.       MM  `MMb.           #
#           MM      MM        MM MMmmdM9   MM MM                ,M  `MM     M  `MN. M    ,M  `MM     MM         VMMP     `YMMNq.   MM    `YMMNq.       #
#           MM      MM.      ,MP MM        MM MM.               AbmmmqMA    M   `MM.M    AbmmmqMA    MM      ,   MM    .     `MM   MM  .     `MM       #
#           MM      `Mb.    ,dP' MM        MM `Mb.     ,'      A'     VML   M     YMM   A'     VML   MM     ,M   MM    Mb     dM   MM  Mb     dM       #
#         .JMML.      `"bmmd"' .JMML.    .JMML. `"bmmmd'     .AMA.   .AMMA.JML.    YM .AMA.   .AMMA.JMMmmmmMMM .JMML.  P"Ybmmd"  .JMML.P"Ybmmd"        #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

####################################################################################################################
#              `7MM"""Yp, `7MM"""YMM  `7MM"""Mq. MMP""MM""YMM   .g8""8q. `7MM"""Mq.`7MMF' .g8"""bgd                #
#                MM    Yb   MM    `7    MM   `MM.P'   MM   `7 .dP'    `YM. MM   `MM. MM .dP'     `M                #
#                MM    dP   MM   d      MM   ,M9      MM      dM'      `MM MM   ,M9  MM dM'       `                #
#                MM"""bg.   MMmmMM      MMmmdM9       MM      MM        MM MMmmdM9   MM MM                         #
#                MM    `Y   MM   Y  ,   MM  YM.       MM      MM.      ,MP MM        MM MM.                        #
#                MM    ,9   MM     ,M   MM   `Mb.     MM      `Mb.    ,dP' MM        MM `Mb.     ,'                #
#              .JMMmmmd9  .JMMmmmmMMM .JMML. .JMM.  .JMML.      `"bmmd"' .JMML.    .JMML. `"bmmmd'                 #
####################################################################################################################

#====================================================================================================#
# BERTopic - "all-mpnet-base-v2"
def Get_BERTopic_mpnet_base(df_cleaned_n_all   = None                                    ,
                            saved_fitted_model = "Saving_BERTopic_mpnet_base_.p"         ,
                            saving_folder      = Path("./Saving_Model")                  ,
                            ):

    # Load text data into a list.
    dateset_n_text_list = df_cleaned_n_all["cleaned_text"].values.tolist()

    if os.path.exists(saving_folder / saved_fitted_model):
        topic_model = pickle.load(open(saving_folder / saved_fitted_model, 'rb'))
    else:
        # Tune the pretrained model.
        topic_model = BERTopic(embedding_model="all-mpnet-base-v2").fit(dateset_n_text_list)

        pickle.dump(topic_model, open(saving_folder / saved_fitted_model, 'wb'))
        
    return topic_model


#====================================================================================================#
# BERTopic - "all-roberta-base-v2"   # Poor Performance !!
def Get_BERTopic_roberta_base(df_cleaned_n_all   = None                                      ,
                              saved_fitted_model = "Saving_BERTopic_roberta_base_.p"         ,
                              saving_folder      = Path("./Saving_Model")                    ,
                              ):

    # Load text data into a list.
    dateset_n_text_list = df_cleaned_n_all["cleaned_text"].values.tolist()

    if os.path.exists(saving_folder / saved_fitted_model):
        topic_model = pickle.load(open(saving_folder / saved_fitted_model, 'rb'))
    else:
        # Tune the pretrained model.
        roberta = TransformerDocumentEmbeddings('roberta-base')
        topic_model = BERTopic(embedding_model = roberta).fit(dateset_n_text_list)
        pickle.dump(topic_model, open(saving_folder / saved_fitted_model, 'wb'))
        
    return topic_model

#====================================================================================================#
# BERTopic - Advanced Customization.
def Get_BERTopic_custiomized(df_cleaned_n_all   = None                                     ,
                             saved_fitted_model = "Saving_BERTopic_custiomized_.p"         ,
                             saving_folder      = Path("./Saving_Model")                   ,
                             ):

    # Load text data into a list.
    dateset_n_text_list = df_cleaned_n_all["cleaned_text"].values.tolist()

    if os.path.exists(saving_folder / saved_fitted_model):
        topic_model = pickle.load(open(saving_folder / saved_fitted_model, 'rb'))
    else:
        from umap import UMAP
        umap_model = UMAP(n_neighbors = 15, n_components = 10, min_dist = 0.0, metric = 'cosine')

        # Tune the pretrained model.
        topic_model = BERTopic(language                = "english"            ,
                               embedding_model         = "all-mpnet-base-v2"  ,
                               calculate_probabilities = True                 , 
                               verbose                 = True                 ,
                               n_gram_range            = (1, 2)               ,
                               #umap_model              = umap_model           ,
                               
                               ).fit(dateset_n_text_list)
        
        pickle.dump(topic_model, open(saving_folder / saved_fitted_model, 'wb'))
        
    return topic_model


#====================================================================================================#
# BERTopic - Original Settings.
def Get_BERTopic_original(df_cleaned_n_all   = None                                  ,
                          saved_fitted_model = "Saving_BERTopic_original_.p"         ,
                          saving_folder      = Path("./Saving_Model")                ,
                          ):

    # Load text data into a list.
    dateset_n_text_list = df_cleaned_n_all["cleaned_text"].values.tolist()

    if os.path.exists(saving_folder / saved_fitted_model):
        topic_model = pickle.load(open(saving_folder / saved_fitted_model, 'rb'))
    else:
        from umap import UMAP
        umap_model = UMAP(n_neighbors = 15, n_components = 10, min_dist = 0.0, metric = 'cosine')
        topic_model = BERTopic(language                  = "english"     , 
                               calculate_probabilities   = True          , 
                               verbose                   = True          ,
                               n_gram_range              = (1, 2)        ,
                               umap_model                = umap_model    ,
                               ).fit(dateset_n_text_list)
        
        pickle.dump(topic_model, open(saving_folder / saved_fitted_model, 'wb'))
        
    return topic_model



#====================================================================================================#
# Add BERTopic to Cleaned Dataframe.
def Add_BERTopic_to_Dataframe(df_cleaned_n_analysis = None  ,
                              topic_model           = None  ,
                              saving_folder         = None  ,
                              saving_file           = None  ,
                              force_redo            = False ,
                              ):

    if os.path.exists(saving_folder / saving_file) and not force_redo:
        df_n_analysis = pd.read_pickle(saving_folder / saving_file)

    else:
        topic_list      = topic_model.topics_
        topic_info_df   = topic_model.get_topic_info()
        topic_name_list = topic_info_df["Name"].values.tolist()
        topic_ID_list   = topic_info_df["Topic"].values.tolist()
        topic_name_dict = {topic_ID_list[i] : topic_name_list[i] for i in range(topic_info_df.shape[0])}
        
    
        df_n_analysis = copy.deepcopy(df_cleaned_n_analysis)
        df_n_analysis["BERTopic_ID"  ] = topic_list
        df_n_analysis["BERTopic_Name"] = [topic_name_dict[i] for i in topic_list]
        df_n_analysis.to_pickle(saving_folder / saving_file)

    return df_n_analysis
















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
def Part_3_Main():

    #====================================================================================================#
    # Args
    Step_code               = "P03A_"
    saved_preproc_dataset_1 = "Saving_Part_3_preproc_dataset_1.p"
    saved_preproc_dataset_2 = "Saving_Part_3_preproc_dataset_2.p"
    reprocess_dataset       = False

    #====================================================================================================#
    # Process Data 
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

    # df_cleaned_1_all.to_csv(path_or_buf = "Saving_Part_3_preproc_dataset_1.csv")
    # df_cleaned_2_all.to_csv(path_or_buf = "Saving_Part_3_preproc_dataset_2.csv")




    #====================================================================================================#
    # Get Sentiment Labels - Implement four different models (trained already)
    df_cleaned_1_analysis = \
        Get_Sentiment_Label_Prediction(df_cleaned_n_all   = df_cleaned_1_all                         ,
                                       saving_file        = "Saving_Part_3_Sentiment_Label_Pred_1.p" ,
                                       saving_folder      = Path("./")                               ,
                                       force_DistilBERT   = False                                    ,
                                       force_Vader        = False                                    ,
                                       Vader_threshold    = 0.42                                     ,
                                       )

    df_cleaned_2_analysis = \
        Get_Sentiment_Label_Prediction(df_cleaned_n_all   = df_cleaned_2_all                         ,
                                       saving_file        = "Saving_Part_3_Sentiment_Label_Pred_2.p" ,
                                       saving_folder      = Path("./")                               ,
                                       force_DistilBERT   = False                                    ,
                                       force_Vader        = False                                    ,
                                       Vader_threshold    = 0.42                                     ,
                                       )

    # print(df_cleaned_1_analysis)
    # print(df_cleaned_2_analysis)

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Comparing Predicted Labels.
    '''
    Comparing_predicted_labels(dataframe_analysis = df_cleaned_1_analysis   ,
                               column_1           = "DistilBert_sentiment"  ,
                               column_2           = "Vader_sentiment_neu"   ,
                               label_1            = "DistilBert"            ,
                               label_2            = "Vader"                 ,
                               dataset_index      = "_1"                    ,
                               )
                               '''

    '''
    Comparing_predicted_labels(dataframe_analysis = df_cleaned_2_analysis   ,
                               column_1           = "DistilBert_sentiment"  ,
                               column_2           = "Vader_sentiment_neu"   ,
                               label_1            = "DistilBert"            ,
                               label_2            = "Vader"                 ,
                               dataset_index      = "_2"                    ,
                               )
                               '''



    #====================================================================================================#
    # Topic Analysis - Apply the BERTopic Model to dateset.
    # Dataset #1
    BERTopic_mpnet_base_1 = \
        Get_BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_1_all                   ,
                                saved_fitted_model = "Saving_BERTopic_mpnet_base_1.p"   ,
                                )

    # Dataset #2
    BERTopic_mpnet_base_2 = \
        Get_BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_2_all                   ,
                                saved_fitted_model = "Saving_BERTopic_mpnet_base_2.p"   ,
                                )

    # Dataset #1
    BERTopic_customized_1 = \
        Get_BERTopic_custiomized(df_cleaned_n_all   = df_cleaned_1_all                  ,
                                 saved_fitted_model = "Saving_BERTopic_customized_1.p"  ,
                                 )
    
    # Dataset #1
    BERTopic_original_1 = \
        Get_BERTopic_original(df_cleaned_n_all   = df_cleaned_1_all                     ,
                              saved_fitted_model = "Saving_BERTopic_original_1.p"       ,
                              )

    print(BERTopic_original_1.get_topic_info().head(50))

    
    # Dataset #1
    BERTopic_original_2 = \
        Get_BERTopic_original(df_cleaned_n_all   = df_cleaned_2_all                       ,
                              saved_fitted_model = "Saving_BERTopic_original_2_0.p"       ,
                              )  
    





                              
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Roberta_base models are found to be outperformed by mpnet_based models.
    '''
    BERTopic_roberta_base_1 = \
        Get_BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_1_all                          ,
                                  saved_fitted_model = "Saving_BERTopic_roberta_base_1.p" ,
                                  )
                                  '''
    
    '''
    BERTopic_roberta_base_2 = \
        Get_BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_2_all                          ,
                                saved_fitted_model   = "Saving_BERTopic_roberta_base_2.p" ,
                                )
                              '''


    #====================================================================================================#
    # Add BERTopic to the cleaned dataframe. 
    df_1_analysis = \
        Add_BERTopic_to_Dataframe(df_cleaned_n_analysis = df_cleaned_1_analysis                   ,
                                  topic_model           = BERTopic_mpnet_base_1                   ,
                                  saving_folder         = Path("./")                              ,
                                  saving_file           = "Saving_Part_3_BERTopic_df_1.p"         ,
                                  )

    df_2_analysis = \
        Add_BERTopic_to_Dataframe(df_cleaned_n_analysis = df_cleaned_2_analysis                   ,
                                  topic_model           = BERTopic_mpnet_base_2                   ,
                                  saving_folder         = Path("./")                              ,
                                  saving_file           = "Saving_Part_3_BERTopic_df_2.p"         ,
                                  )

    print("\n\n" + "="*125 + "\nAdd BERTopic to the cleaned dataset #1: " )
    beautiful_print(df_1_analysis)

    print("\n\n" + "="*125 + "\nAdd BERTopic to the cleaned dataset #2: " )
    beautiful_print(df_2_analysis)

    



    return 






if __name__ == "__main__":
    Part_3_Main()
    
    #test_Get_11_Sentiment_labels()





















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
