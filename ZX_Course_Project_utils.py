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
# import
#--------------------------------------------------#
import os 
import sys
import os.path
from sys import platform
from pathlib import Path
#--------------------------------------------------#
import re
import sys
import time
import copy
import math
import html
import scipy
import pickle
import random
import argparse
import subprocess
import numpy as np
import pandas as pd

#--------------------------------------------------#
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

#--------------------------------------------------#
# import xmltodict
#--------------------------------------------------#
from timeit import timeit
#--------------------------------------------------#
import urllib
import xml.etree.ElementTree as ET
#--------------------------------------------------#
from datasets import load_metric
from urllib.request import urlopen
#--------------------------------------------------#
from sklearn.model_selection import train_test_split
#--------------------------------------------------#
from transformers import RobertaForSequenceClassification, RobertaTokenizerFast, Trainer, TrainingArguments
#--------------------------------------------------#





#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    `7MM"""Mq.`7MM"""Mq. `7MM"""YMM  `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd      #
#      MM   `MM. MM   `MM.  MM    `7    MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M      #
#      MM   ,M9  MM   ,M9   MM   d      MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       `      #
#      MMmmdM9   MMmmdM9    MMmmMM      MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM               #
#      MM        MM  YM.    MM   Y  ,   MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF'    #
#      MM        MM   `Mb.  MM     ,M   MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM      #
#    .JMML.    .JMML. .JMM.JMMmmmmMMM .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#


# For clearly print a dataframe. 
def beautiful_print(df): # Print the DataFrame obtained (May NOT work properly in jupyter notebook).
    # Print the dataset in a well-organized format.
    with pd.option_context('display.max_rows'       , 10   , 
                           'display.min_rows'       , 10   , 
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


###################################################################################################################
#                         .g8"""bgd `7MMF'      `7MM"""YMM        db     `7MN.   `7MF'                            #
#                       .dP'     `M   MM          MM    `7       ;MM:      MMN.    M                              #
#                       dM'       `   MM          MM   d        ,V^MM.     M YMb   M                              #
#                       MM            MM          MMmmMM       ,M  `MM     M  `MN. M                              #
#                       MM.           MM      ,   MM   Y  ,    AbmmmqMA    M   `MM.M                              #
#                       `Mb.     ,'   MM     ,M   MM     ,M   A'     VML   M     YMM                              #
#                         `"bmmmd'  .JMMmmmmMMM .JMMmmmmMMM .AMA.   .AMMA.JML.    YM                              #
###################################################################################################################


# Extra Options for preprocessing text data.
def clean_text_preproc(text):
    # List of extra functions for dealing with raw text. 
    #--------------------------------------------------#
    # (1) Replace newlines with spaces.
    def remove_whitespace(s):
        # return re.sub(r"\n{1,}|\\n{1,}|\r{1,}|\\r{1,}|\t{1,}|\\t{1,}", " ", s)
        return re.sub(r'\s', " ", s)
    # (2) Unescape html and replace html character codes with ascii equivalent.
    def unescape_html(s):
        return html.unescape(s)
    # (3) remove URLs.
    def remove_urls(s):
        return re.sub(r"\b(http:\/\/|https:\/\/|www\.)\S+", "", s)
    # (4) remove duplicate spaces.
    def remove_duplicate_spaces(s):
        return re.sub(r" {2,}", " ", s)
    # (5) 
    def remove_tags(text):
        text = re.sub(r'@[A-Za-z0-9_:]+', '', text)  
        text = re.sub(r'#(\S+)', r'\1', text)        
        text = re.sub(r'^RT ', '', text)             
        return text
    #--------------------------------------------------#
    text = remove_whitespace       (text)
    text = unescape_html           (text)
    text = remove_urls             (text)
    text = remove_duplicate_spaces (text)
    text = remove_tags             (text)
    text = remove_whitespace       (text)
    text = remove_duplicate_spaces (text)
    return text

#====================================================================================================#
# Remove unicode characters from the string.
def remove_unicode(str_x):
    return str_x.encode("ascii", "ignore").decode()

#====================================================================================================#
# Clean the data for Language Models (For Later Use.)
def clean_text_LM(text):
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'@[A-Za-z0-9_:]+', '', text)          # Remove User tags
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    #text = re.sub(r'[^A-Za-z0-9 ]+', '', text)          # Remove non-alphanumeric characters
    text = re.sub(r'#(\S+)', r'\1', text)                # Remove "#" in Hashtags.
    text = re.sub(r'^RT ', '', text)                     # Remove Retweet

    def replacer(match):
        if match.group(1) is not None:
            return '{} '.format(match.group(1))
        else:
            return ' {}'.format(match.group(2))

    rx = re.compile(r'^(\W+)|(\W+)$')
    text = " ".join([rx.sub(replacer, word) for word in text.split()])
    return remove_unicode(text.strip()).lower()

#====================================================================================================#
# Clean the data for Sentiment Classifer to be trained.
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)                    # Remove HTML tags
    text = re.sub(r'@[A-Za-z0-9_:]+', '', text)          # Remove user tags
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)           # Remove non-alphanumeric characters
    text = re.sub(r'#(\S+)', r'\1', text)                # Remove "#" in Hashtags.
    text = re.sub(r'^RT ', '', text)                     # Remove Retweet
    return remove_unicode(text.strip()).lower()

#====================================================================================================#
# Check the string is fully ASCII chars or not. (for test use only)
def is_ASCII_only_string(s):
    return not bool(re.search('[^\x00-\x7F]+', s))





####################################################################################################################
# `7MM"""YMM  `7MN.   `7MF' .g8"""bgd      `7MM"""Yb. `7MM"""YMM MMP""MM""YMM `7MM"""YMM    .g8"""bgd MMP""MM""YMM #
#   MM    `7    MMN.    M .dP'     `M        MM    `Yb. MM    `7 P'   MM   `7   MM    `7  .dP'     `M P'   MM   `7 #
#   MM   d      M YMb   M dM'       `        MM     `Mb MM   d        MM        MM   d    dM'       `      MM      #
#   MMmmMM      M  `MN. M MM                 MM      MM MMmmMM        MM        MMmmMM    MM               MM      #
#   MM   Y  ,   M   `MM.M MM.    `7MMF'      MM     ,MP MM   Y  ,     MM        MM   Y  , MM.              MM      #
#   MM     ,M   M     YMM `Mb.     MM        MM    ,dP' MM     ,M     MM        MM     ,M `Mb.     ,'      MM      #
# .JMMmmmmMMM .JML.    YM   `"bmmmdPY      .JMMmmmdP' .JMMmmmmMMM   .JMML.    .JMMmmmmMMM   `"bmmmd'     .JMML.    #
####################################################################################################################

def Load_spacy():
    # Check the string is English or NOT. (using the ``spaCy`` package)
    # Can also use `TextBlob` or `Pycld2` but I dont want to use google API.
    print("\n" + "="*80)
    print("spaCy is a poackage with tons of compatability issues.")
    print("``spaCy`` is used to detect language. Need to, \n(1)install ``spaCy`` \n(2)Download models via")
    print("    python -m spacy download en_core_web_lg")
    print("    python -m spacy download en_core_web_sm")
    # !pip install spacy
    # !pip install spacy_langdetect
    # !python -m spacy download en_core_web_lg
    # !python -m spacy download en_core_web_sm
    # !python -m spacy download en_core_web_trf
    #--------------------------------------------------#
    # Prepares
    import spacy
    from spacy.language import Language
    from spacy_langdetect import LanguageDetector
    nlp = spacy.load("en_core_web_lg")
    def get_lang_detector(nlp, name):
        return LanguageDetector()
    # Language Factory cannot be loaded multiple times for no reason.
    # Therefore, have to use try/except block.
    try:
        Language.factory("language_detector", func = get_lang_detector)
    except:
        pass
    nlp.add_pipe('language_detector', last = True)
    return nlp

#--------------------------------------------------#
# Define a function for identifying language (whether English).
def is_English(text_str):
    doc = nlp(text_str)
    detect_language = doc._.language["language"] 
    score           = doc._.language["score"]
    return (detect_language == "en"), detect_language, score

def is_English_batch_process(text_list):
    ifEnglish_list = []
    dtct_lang_list = []
    score_list = []
    for idx, doc in enumerate(nlp.pipe(text_list, n_process = 4, batch_size = 2000)):
        lang_detect_result  =  doc._.language
        detect_language     =  lang_detect_result["language"] 
        score               =  lang_detect_result["score"]
        ifEnglish_list.append(detect_language == "en")
        dtct_lang_list.append(detect_language)
        score_list    .append(score)
    return ifEnglish_list, dtct_lang_list, score_list


# All-Caps Text will be detected as German.
# Very short text will be detected as random language. 
# print(is_English(  "Some English Text are often FALSEly identified" ))
# print(is_English(  "happy birthday"                                 ))  
# print(is_English(  "happy birthday to you my little sweetie"        ))
# print(is_English(  "Le client est très important merci"             ))
# print(is_English(  "WHAT FUCKING LANGUAGE IS MY FUCKING PHONE IN"   ))  




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#    `7MM"""Mq.`7MM"""Mq. `7MM"""YMM  `7MM"""Mq.`7MM"""Mq.   .g8""8q.     .g8"""bgd `7MM"""YMM   .M"""bgd  .M"""bgd `7MMF'`7MN.   `7MF' .g8"""bgd      #
#      MM   `MM. MM   `MM.  MM    `7    MM   `MM. MM   `MM..dP'    `YM. .dP'     `M   MM    `7  ,MI    "Y ,MI    "Y   MM    MMN.    M .dP'     `M      #
#      MM   ,M9  MM   ,M9   MM   d      MM   ,M9  MM   ,M9 dM'      `MM dM'       `   MM   d    `MMb.     `MMb.       MM    M YMb   M dM'       `      #
#      MMmmdM9   MMmmdM9    MMmmMM      MMmmdM9   MMmmdM9  MM        MM MM            MMmmMM      `YMMNq.   `YMMNq.   MM    M  `MN. M MM               #
#      MM        MM  YM.    MM   Y  ,   MM        MM  YM.  MM.      ,MP MM.           MM   Y  , .     `MM .     `MM   MM    M   `MM.M MM.    `7MMF'    #
#      MM        MM   `Mb.  MM     ,M   MM        MM   `Mb.`Mb.    ,dP' `Mb.     ,'   MM     ,M Mb     dM Mb     dM   MM    M     YMM `Mb.     MM      #
#    .JMML.    .JMML. .JMM.JMMmmmmMMM .JMML.    .JMML. .JMM. `"bmmd"'     `"bmmmd'  .JMMmmmmMMM P"Ybmmd"  P"Ybmmd"  .JMML..JML.    YM   `"bmmmdPY      #
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#























































































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
