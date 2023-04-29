# <center> <font size="20">  Sentiment Analysis  </font> </center>

we leverage advanced deep learning methodologies to conduct a robust sentiment analysis on social media conversations surrounding the Ukraine-Russia conflict. Our methodology incorporates the training of numerous sentiment classification models, starting with four non-neural network baseline models. (i.e., (1) logistic regression, (2) decision tree classifier, (3) Gaussian Na¨ıve Bayes and (4) XGBoost classifier), using a dataset that contains over 550,000 tweets that have had their sentiments already analyzed and recorded as binary values 0 (negative) and 1 (positive). The data processing pipelines for training baselines are shown below. 

<p align="center">
  <img width="1000"  src="https://user-images.githubusercontent.com/47986787/235311492-c96a757b-3038-42b3-8d09-30fc072cf2dd.png">
</p>

We have also trained a 3-layer multilayer perceptron (MLP) classifier and a DistilBERT-based classifier based on the state-of-the-art NLP techniques. Apart from these binary sentiment classifiers, we also train a multilabel classification model that is capable of predicting the 8 primary bipolar emotions (joy versus sadness, anger versus fear, trust versus disgust, and surprise versus anticipation) from Robert Plutchik’s wheel of emotions shown below. 

<p align="center">
  <img width="600"  src="https://user-images.githubusercontent.com/47986787/235311766-7ddfcc15-7475-4455-9b84-ebee53bb1162.png">
</p>

We utilize tweets from the SemEval2018 Task 1E-c emotion classification dataset to perform fine-tuning on the pretrained DistilBERT model. The dataset also contains binary labels 0 (negative) and 1 (positive) for optimism, pessimism and love. These labels are used to train 3 additional DistilBERT-based classifiers for predicting the existence of these emotions. Models trained by others are also used as comparisons to our models, including VADER (Valence Aware Dictionary and sentiment Reasoner) and a pre-trained DistilBERT-based threefold sentiment classification model (from https://huggingface.co/Souvikcmsa/ SentimentAnalysisDistillBERT).

These machine learning-based sentiment analysis models are then implemented to identify sentiments and emotions for scrapped comments and posts related to the Russian Ukraine War from Reddit, a social news aggregation and discussion platform. Sentiment labels predicted by different models are then compared in order to validate the performances of models that we have trained. We are able to demonstrate that the deep learning-based sentiment classifier we have trained is able to achieve comparable results to models commonly used in industry as shown by the confusion matrix below. 

<p align="center">
  <img width="600"  src="https://user-images.githubusercontent.com/47986787/235312144-f5f51f06-b5e2-426b-a878-15b8525d1e76.png">
</p>

In order to identify factors/reasons/topics that drive sentiment, we utilize [BERTopic](https://github.com/MaartenGr/BERTopic) to discover the most influential topics. We analyze the sentiments of each comment under these hot topics and study the public opinion regarding the Russia-Ukraine War as well as other related international events. The analysis can be used to explore the most concerning topics and provide suggestions to the Ukrainian government and international NGOs. Some results are shown below, 

### Pray for Ukraine
![image](https://user-images.githubusercontent.com/47986787/235312230-54b1bb97-6c1b-4620-aeb8-a8db41831357.png)

![image](https://user-images.githubusercontent.com/47986787/235312223-608b8ef3-6fd4-4296-91fd-d77474387f9e.png)


Through a comprehensive analysis of the comments scrapped from Twitter and Reddit, we explore public sen- timent towards Russia and Ukraine, perceptions of President Zelensky, concerns about casualties, debates over re- sponsibility for the conflict, and dissatisfaction with NATO’s involvement. This analysis is intended to serve as a consulting tool for the Ukrainian government and international NGOs to analyze how Ukraine is perceived inter- nationally and provide valuable recommendations to improve Ukraine’s international presence and image. All our models and results are available through https://github.com/Zhiqing-Xu/SentimentAnalysis.





