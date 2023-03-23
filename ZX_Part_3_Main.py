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


#--------------------------------------------------#
from ZX_Course_Project_utils import *



#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#               `7MM"""Yb. `7MM"""YMM   .M"""bgd   .g8"""bgd `7MM"""Mq. `7MMF'`7MM"""Mq. MMP""MM""YMM `7MMF' .g8""8q. `7MN.   `7MF'                    #
#                 MM    `Yb. MM    `7  ,MI    "Y .dP'     `M   MM   `MM.  MM    MM   `MM.P'   MM   `7   MM .dP'    `YM. MMN.    M                      #
#                 MM     `Mb MM   d    `MMb.     dM'       `   MM   ,M9   MM    MM   ,M9      MM        MM dM'      `MM M YMb   M                      #
#                 MM      MM MMmmMM      `YMMNq. MM            MMmmdM9    MM    MMmmdM9       MM        MM MM        MM M  `MN. M                      #
#                 MM     ,MP MM   Y  , .     `MM MM.           MM  YM.    MM    MM            MM        MM MM.      ,MP M   `MM.M                      #
#                 MM    ,dP' MM     ,M Mb     dM `Mb.     ,'   MM   `Mb.  MM    MM            MM        MM `Mb.    ,dP' M     YMM                      #
#               .JMMmmmdP' .JMMmmmmMMM P"Ybmmd"    `"bmmmd'  .JMML. .JMM.JMML..JMML.        .JMML.    .JMML. `"bmmd"' .JML.    YM                      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

# Part 3 - Instruction
'''
Based on your sentiment analysis results, 
    (especially looking at posts/tweets with negative and positive sentiment, 
    as posts/tweets with neutral sentiment are less informative), 
you will need to IDENTIFY factors/reasons/TOPICS that drive sentiment. 

Those are factors (reasons, TOPICS) that explain sentiment and can be used 
for decision making and recommendations in Part 4. 

You can use any Natural Language Processing models for this part of the project 
(use existing models or develop your own).
'''

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
def Clean_data(df_raw_n, num_words_lb): # num_words_lb is the minimum number of words in one sentence.

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

# Sentiment Label Prediction via Pretrained Distilled BERT and Vader.

#====================================================================================================#
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



#====================================================================================================#
# Sentiment Label Prediction via Pretrained Vader.

# from typing import Optional, Union, Tuple, Type, Sequence, List, Set, Any, TextIO, IO
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


#====================================================================================================#
# Sentiment Label Prediction Savings.
def Get_Sentiment_Label_Prediction(df_cleaned_n_all   = None                              ,
                                   saving_file        = "Saving_Sentiment_Label_Pred_.p"  ,
                                   saving_folder      = Path("./")                        ,
                                   force_DistilBERT   = False                             ,
                                   force_Vader        = False                             ,
                                   Vader_threshold    = 0.05                              ,
                                   ):

    if os.path.exists(saving_folder / saving_file):
        df_cleaned_n_analysis = pd.read_pickle(saving_folder / saving_file)

        if force_DistilBERT:
            dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
            df_cleaned_n_analysis["DistilBertLabel"] = Get_DistillBERT_sentiment_labels(dataset_n_text)[0]

        if force_Vader:
            df_cleaned_n_analysis["Vader_sentiment"] = \
                Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist() , 
                                 threshold = 0.0                                       )
            
            df_cleaned_n_analysis["Vader_sentiment_neu"] = \
                Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist() , 
                                 threshold = Vader_threshold                           )

        label_list = df_cleaned_n_analysis["DistilBertLabel"].values.tolist()
        df_cleaned_n_analysis["DistilBert_sentiment"] = ["neg" if l==0 else "neu" if l==1 else "pos" for l in label_list]

        df_cleaned_n_analysis.to_pickle(saving_folder / saving_file)

    else:
        # Make a copy.
        df_cleaned_n_analysis = copy.deepcopy(df_cleaned_n_all)

        #====================================================================================================#
        # DistilBert Sentiment Labels. 
        dataset_n_text = df_cleaned_n_all["cleaned_text"].values.tolist()
        df_cleaned_n_analysis["DistilBertLabel"] = Get_DistillBERT_sentiment_labels(dataset_n_text)[0]
        label_list = df_cleaned_n_analysis["DistilBertLabel"].values.tolist()
        df_cleaned_n_analysis["DistilBert_sentiment"] = ["neg" if l==0 else "neu" if l==1 else "pos" for l in label_list]
        #====================================================================================================#
        # Vader Sentiment Labels. 
        # Use Vader to predict sentiment labels for this dataset. 
        df_cleaned_n_analysis["Vader_sentiment"] = Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist(), threshold = 0.00)
        # Use Vader to predict sentiment labels for this dataset. 
        df_cleaned_n_analysis["Vader_sentiment_neu"] = Vader_Prediction(df_cleaned_n_analysis["cleaned_text"].values.tolist(), threshold = Vader_threshold)
        # Save.
        df_cleaned_n_analysis.to_pickle(saving_folder / saving_file)
    return df_cleaned_n_analysis





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
def Get_BERTopic_mpnet_base(df_cleaned_n_all   = None                             ,
                            saved_fitted_model = "Saving_BERTopic_mpnet_base_.p"  ,
                            saving_folder      = Path("./Saving_Model")           ,
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
# BERTopic - "all-roberta-base-v2"
def Get_BERTopic_roberta_base(df_cleaned_n_all   = None                               ,
                              saved_fitted_model = "Saving_BERTopic_roberta_base_.p"  ,
                              saving_folder      = Path("./Saving_Model")             ,
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
# 


















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
    Step_code               = "P03A_"
    saved_preproc_dataset_1 = "Saving_preproc_dataset_1.p"
    saved_preproc_dataset_2 = "Saving_preproc_dataset_2.p"
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

    # df_cleaned_1_all.to_csv(path_or_buf = "Saving_preproc_dataset_1.csv")
    # df_cleaned_2_all.to_csv(path_or_buf = "Saving_preproc_dataset_2.csv")

    #====================================================================================================#
    # Topic Analysis.

    # Dataset #1
    BERTopic_mpnet_base_1 = \
        Get_BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_1_all                  ,
                                saved_fitted_model = "Saving_BERTopic_mpnet_base_1.p"  ,
                                )
    
    # Dataset #2

    BERTopic_mpnet_base_2 = \
        Get_BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_2_all                  ,
                                saved_fitted_model = "Saving_BERTopic_mpnet_base_2.p"  ,
                                )

    


    #====================================================================================================#
    # Roberta_base models are found to be outperformed by mpnet_based models.

    '''
    BERTopic_roberta_base_1 = \
        Get_BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_1_all                    ,
                                  saved_fitted_model = "Saving_BERTopic_roberta_base_1.p"  ,
                                  )
    

    BERTopic_roberta_base_2 = \
    Get_BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_2_all                    ,
                              saved_fitted_model = "Saving_BERTopic_roberta_base_2.p"  ,
                              )
                              '''




    #====================================================================================================#
    # Get Sentiment Labels.
    df_cleaned_1_analysis = \
        Get_Sentiment_Label_Prediction(df_cleaned_n_all   = df_cleaned_1_all                   ,
                                       saving_file        = "Saving_Sentiment_Label_Pred_1.p"  ,
                                       saving_folder      = Path("./")                         ,
                                       force_DistilBERT   = False                              ,
                                       force_Vader        = False                              ,
                                       Vader_threshold    = 0.025                              ,
                                       )

    df_cleaned_2_analysis = \
        Get_Sentiment_Label_Prediction(df_cleaned_n_all   = df_cleaned_2_all                   ,
                                       saving_file        = "Saving_Sentiment_Label_Pred_2.p"  ,
                                       saving_folder      = Path("./")                         ,
                                       force_DistilBERT   = False                              ,
                                       force_Vader        = False                              ,
                                       Vader_threshold    = 0.025                              ,
                                       )
    # print(df_cleaned_1_analysis)
    # print(df_cleaned_2_analysis)




    #====================================================================================================#
    # Percentage HeatMap
    # Compare the two predictions.
    num_classes = 3
    output_file_header = "Part2"
    plot_name = "Vader Prediction vs. DistilBERT Prediction"
    results_sub_folder = Path("./")
    DBERT_sentiments_1   = df_cleaned_1_analysis["DistilBert_sentiment"] .values.tolist()
    Vader_sentiments_1   = df_cleaned_1_analysis["Vader_sentiment_neu"]  .values.tolist()

    # print(DBERT_sentiments_1)
    # print(Vader_sentiments_1)

    font = {'family' : "Times New Roman"}
    plt.rc('font', **font)
    #--------------------------------------------------#
    cm = confusion_matrix(Vader_sentiments_1, DBERT_sentiments_1)
    cm = cm/len(Vader_sentiments_1) 
    confusion_matrix_df = pd.DataFrame(cm) #.rename(columns=idx2class, index=idx2class)
    #--------------------------------------------------#
    fig = plt.figure(num=None, figsize=(16, 12.8), dpi=80, facecolor='w', edgecolor='k')
    ax = sns.heatmap(confusion_matrix_df, 
                        annot      = True             , 
                        fmt        = ".3f"            , 
                        cmap       = "magma"          , 
                        vmin       = 0.15             , 
                        vmax       = 0.30             , 
                        center     = 0.22             , 
                        cbar_kws   = {"shrink": .82}  , 
                        linewidths = 0.1              , 
                        linecolor  = 'gray'           , 
                        annot_kws  = {"fontsize": 30} , 
                        )

    ax.set_xlabel('DistilBERT Predicted Class', fontsize = 32)
    ax.set_ylabel('Vader Predicted Class', fontsize = 32)
    ax.set_title("Confusion Matrix - " \
                + " \n (Vader Prediction vs. DistilBERT Prediction)", fontsize = 32)
    ax.xaxis.set_ticklabels([["Negative", "Neutral", "Positive"][i] for i in range(num_classes)], fontsize = 32) 
    ax.yaxis.set_ticklabels([["Negative", "Neutral", "Positive"][i] for i in range(num_classes)], fontsize = 32) 
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.show()
    #--------------------------------------------------#
    saving_name = output_file_header + "_HeatMapCM_" + plot_name +"_percentage.png"
    saving_name = saving_name 
    fig.savefig(results_sub_folder / saving_name , dpi = 1000 )
    mpl.rcParams.update(mpl.rcParamsDefault)





    #====================================================================================================#
    # BERTopic_mpnet_base on Dataset #1

    # Manually select interested topics and perform analysis.
    print(BERTopic_mpnet_base_1.get_topic_info().head(50))
    print()

    Dataset_1_topics_dict = { 0 : ["0_nuclear_nukes_nuke_launch"                  , "Nuke and Nuclear War"                ],  
                              2 : ["2_ukraine_will_they_russia"                   , "Ukraine-Russia issues"               ],  
                              3 : ["3_finland_sweden_nato_finnish"                , "Finland, Sweden and NATO"            ],  
                              4 : ["4_orban_hungary_hungarian_hungarians"         , "Orban and Hungary"                   ],  
                              6 : ["6_moldova_romania_transnistria_moldovan"      , "Moldova, Romania and Transnistria"   ],  
                              7 : ["7_barrel_bore_center_lathe"                   , "Military Industry"                   ],  
                              8 : ["8_surrender_casualties_civilians_civilian"    , "Surrender, Casualties and Civilians" ],  
                              9 : ["9_germany_gas_german_bundeswehr"              , "Germany"                             ],  
                             10 : ["10_threat_provoking_russia_threats"           , "Russia Provoking Threats"            ],  
                             11 : ["11_zelensky_zelenskyy_munich_security"        , "Zelensky"                            ],  
                             12 : ["12_putin_he_his_him"                          , "Putin"                               ],  
                             13 : ["13_nato_russia_war_join"                      , "Nato and Russia Relation"            ],  
                             14 : ["14_uk_eu_europe_france"                       , "UK, EU, Europe and France"           ],  
                             16 : ["16_nato_members_alliance_defensive"           , "Nato Alliance Defense"               ],  
                             17 : ["17_lie_lying_truth_media"                     , "Media Truth or Lie"                  ],  
                             18 : ["18_trump_asset_his_putin"                     , "Trump and Putin"                     ],  
                             19 : ["19_reactor_reactors_chernobyl_water"          , "Chernobyl Reactors"                  ],  
                             20 : ["20_mercenaries_mercenary_conflict_armed"      , "Mercenary"                           ],  
                             21 : ["21_baltic_baltics_lithuania_states"           , "Three Baltic Countries"              ],  
                             22 : ["22_himars_missiles_range_system"              , "Firearms and Weapons"                ],  
                             23 : ["23_ukraine_pray_my_peace"                     , "Pray Peace for Ukraine"              ],  
                             24 : ["24_lose_war_putin_winning"                    , "Putin"                               ],  
                             27 : ["27_belarus_continent_forget_center"           , "Belarus"                             ],  
                             29 : ["29_economy_economical_economics_shortages"    , "Economy"                             ],  
                             30 : ["30_bully_you_violence_fight"                  , "Violence"                            ],  
                             34 : ["34_explosion_shot_tank_explosions"            , "Firearms and Weapons"                ],  
                             36 : ["36_china_taiwan_hong_kong"                    , "China Taiwan Issue"                  ],  
                             37 : ["37_nato_ukraine_troops_border"                , "Nato and Ukraine Troops"             ],  
                             39 : ["39_biden_trump_republicans_he"                , "American Politicians"                ],  
                             41 : ["41_he_putin_ukraine_him"                      , "Putin"                               ],  
                             42 : ["42_language_russian_ukrainian_speak"          , "Ukraine-Russia issues"               ],  
                             43 : ["43_minsk_ukraine_russian_russians"            , "Belarus"                             ],  
                             44 : ["44_republican_republicans_poorly_party"       , "Republican"                          ],  
                             45 : ["45_phosphorus_incendiary_smoke_banned"        , "Firearms and Weapons"                ],  
                             47 : ["47_switchblade_artillery_armor_fire"          , "Firearms and Weapons"                ],  
                             }

    # BERTopic_mpnet_base_1.get_topic(0) # Get Keywords in Topic #0


    print(BERTopic_mpnet_base_1.topics_[:10]) # Topic ID for the first 10 sentences in the document.
    print()

    #topic_model.visualize_hierarchy(top_n_topics=50)






    print(BERTopic_mpnet_base_1.get_document_info(BERTopic_mpnet_base_1))



    Dataset_1_keyword_list = ["Nuke", "Ukraine", "Russia", "Hungary", "Germany", "Nato", "Putin", "Zelensky", "Biden", "Trump", "Firearms"]
    similar_topics, similarity = BERTopic_mpnet_base_1.find_topics("vehicle", top_n = 5)
    print(similar_topics)


































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
