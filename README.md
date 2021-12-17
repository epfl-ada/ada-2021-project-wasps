# Detect profile of the speaker based on deep learning

## Abstract

While performing analysis of Quotebank data we found out that around 34% of quotations don't have assigned speakers to it (1.8 million out of 5.2 million in file quotes-2020.json). Our goal is to answer the following question: if we cannot determine the exact author of a quotation, what other information can we get from it?

In that work we would like to extract additional information about known authors parsing information from Wikipedia and use it to describe unknown authors thereby reducing their obscurity. With the data and labels, we trained several models and verified the functionalities, then predicted the features of the quotations that are not assigned speakers in Quotebank. Also, we did some analysis on the outcomes and explored the relationships between different features, as well as tried to understand the mechanism of the prediction.

## Repository structure

```data```: contains parsed data that is split by features.

```figures```: contains plots that we are using for the README.

```notebooks```: 

```scripts```: contain python file with our classification model.

## Contribution

_Pawel Mlyniec_: collection data from wikipedia dumps, parsing all the data, sptilling the data into feature categoties, adapting data format for the usage.

_Sofia Blinova_: creation classification model, writing training, validation and prediction piplines, result analyzation.

_Ekaterina Trimbach_: creation most of the visualization, predicted labels and topic analization.

_Wei Shi_: Result analysis on preliminary experiment, Website configuration, models training and records, write huge part of the report.

_All together_: work on the final report.

