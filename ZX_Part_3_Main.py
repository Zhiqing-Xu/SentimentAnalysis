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


#--------------------------------------------------#
from sklearn.model_selection import train_test_split

#--------------------------------------------------#
# Pretrained Models.
# Topic Models
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings

# Language Models.
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments





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
def BERTopic_mpnet_base(
                        df_cleaned_n_all   = None                             ,
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
        pass

    print(topic_model.get_topic_info().head(10))

    return


#====================================================================================================#
# BERTopic - "all-mpnet-base-v2"
def BERTopic_roberta_base(
                          df_cleaned_n_all   = None                               ,
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
        pass

    print(topic_model.get_topic_info().head(10))

    return














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
    # Process Data 


    BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_1_all                  ,
                        saved_fitted_model = "Saving_BERTopic_mpnet_base_1.p"  ,
                       )
    

    BERTopic_mpnet_base(df_cleaned_n_all   = df_cleaned_2_all                  ,
                        saved_fitted_model = "Saving_BERTopic_mpnet_base_2.p"  ,
                       )

    
    BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_1_all                    ,
                          saved_fitted_model = "Saving_BERTopic_roberta_base_1.p"  ,
                          )
    
    BERTopic_roberta_base(df_cleaned_n_all   = df_cleaned_2_all                    ,
                          saved_fitted_model = "Saving_BERTopic_roberta_base_2.p"  ,
                          )











































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
