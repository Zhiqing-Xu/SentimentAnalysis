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

# import xmltodict
#--------------------------------------------------#
from timeit import timeit

#--------------------------------------------------#
import urllib
import xml.etree.ElementTree as ET
#--------------------------------------------------#


#--------------------------------------------------#
from datasets import load_metric
from urllib.request import urlopen

#--------------------------------------------------#
from sklearn.model_selection import train_test_split

#--------------------------------------------------#
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments

#--------------------------------------------------#

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





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    `7MM"""Mq.`7MM"""Mq. `7MM"""YMM  `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd      #
#      MM   `MM. MM   `MM.  MM    `7    MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M      #
#      MM   ,M9  MM   ,M9   MM   d      MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       `      #
#      MMmmdM9   MMmmdM9    MMmmMM      MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM               #
#      MM        MM  YM.    MM   Y  ,   MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF'    #
#      MM        MM   `Mb.  MM     ,M   MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM      #
#    .JMML.    .JMML. .JMM.JMMmmmmMMM .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#

#%%
# Load data
#====================================================================================================#
data_folder = Path("./data_folder")
data_file   = "sentiment_analysis.csv"

# Load data
#====================================================================================================#
# For clearly print a dataframe. 
def beautiful_print(df): # Print the DataFrame obtained.
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows'       , 20   , 
                           'display.min_rows'       , 20   , 
                           'display.max_columns'    , 15   , 
                           "display.max_colwidth"   , 120 ,
                           "display.width"          , None ,
                           "expand_frame_repr"      , True ,
                           "max_seq_items"          , None , ):  # more options can be specified
        # Once the display.max_rows is exceeded, 
        # the display.min_rows options determines 
        # how many rows are shown in the truncated repr.
        print(df)
    return 

#====================================================================================================#
# Load data with "ISO-8859-1" encoding.
df_raw = pd.read_csv(filepath_or_buffer   =   data_folder/data_file             ,
                     header               =   0                                 , 
                     #index_col           =   0                                 ,
                     encoding             =   "ISO-8859-1"                      ,
                     sep                  =   ','                               ,  
                     low_memory           =   False                             , )

# Print dimension of the raw data.
print("\nRaw dataset printed below. ")
print("df_raw.shape: ", df_raw.shape) # 298 column
beautiful_print(df_raw)

def remove_unicode(str_x):
    return str_x.encode("ascii", "ignore").decode()

def clean_text_LM(text):
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'@[A-Za-z0-9_:]+', '', text)          # Remove User tags
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    #text = re.sub(r'[^A-Za-z0-9 ]+', '', text)          # Remove non-alphanumeric characters
    text = re.sub(r'#(\S+)', r'\1', text)                # Remove "#" in Hashtags.
    text = re.sub(r'^RT ', '', text)                     # Remove Retweet
    return remove_unicode(text.strip())

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'@[A-Za-z0-9_:]+', '', text)            # Remove user tags
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)           # Remove non-alphanumeric characters
    text = re.sub(r'#(\S+)', r'\1', text)                # Remove "#" in Hashtags.
    text = re.sub(r'^RT ', '', text)                     # Remove Retweet
    return remove_unicode(text.strip()).lower()


df_raw['cleaned_text'] = df_raw['text'].apply(clean_text_LM)

df_cleaned = copy.deepcopy(df_raw).drop(columns=["ID", "text"])
df_cleaned.insert(len(df_cleaned.columns)-1, "label", df_cleaned.pop('label') ) 

print("\nProcessed dataset printed below. ")
print("df_cleaned.shape: ", df_cleaned.shape) # 298 column
beautiful_print(df_cleaned)

print(df_cleaned['label'].value_counts())


#%%

#====================================================================================================#









































































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
