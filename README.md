# Detect profile of the speaker based on deep learning
## Abstract
While performing analysis of Quotebank data we found out that around 34% of quotations don't have assigned speakers to it (1.8 million out of 5.2 million in file quotes-2020.json). Our goal is to answer the following question: if we cannot determine the exact author of a quotation, what other information can we get from it?

We aim at finding more information about already known authors from additional datasets such as Wikipedia and use them to train Deep Learning model to predict these new features for unknown classes.

Another sub-task is to categorise quotations into categories extracted from newspaper urls. We choose the few most popular newspaper sections (NYT) and assign other quotations to them with DistillBERT.


Afterwards we will  analyse behaviour by groups, which could unravel many interesting patterns.
## Research Questions
* Can we predict the topic of quotation (based on the topic in which it appears in the newspaper)?
* Is it possible to determin basic information, such as sex, age, occupation, place of birth, children and political party, from quotation?
* What are the most popular topics in each subgroup?
* What is the sentiment of quotation in each group?
* Are there any patterns between groups and quotations?

## Proposed additional datasets
We will use Wikipedia datasets to enrich our data. Especially we will base on additional metadata collected by ADA team which consist of additional labels as:
* date of birth
* nationality
* gender
* ethnic group
* party
* academic degree
* religion
### Topics analysing
During our analysis we found that the proportions of 10 the most fasmous topics in New Your Times is :
![Screenshot](figures/topics_proportions.png)

## Methods
For data preprocessing we use TODO
### Data Preprocessing
For preprocessing, we exctracted the data of the most popular topics classes (top 1 to top 10 in our experiment) with labels as the dataset for training and test. The datasets are stored on google drive and we load them by urls, the structure of sample ```['qoutation', 'label', 'label_num']```
We use distilBERT from Hugging Face to predict new features based on quotations. 

We will use TODO to final analyses.
## Proposed timeline
* 26.11 - DistilBERT finetuned to predict categories
* 26.11 - All additional labels will be added to base dataset
* 03.12 - Finetune DestilBert to new labels
* 17.12 - Create github page with all analysis of group patterns 

## Organization within the team
Folowing people have this assigments:
* Sofia - Apply DistilBERT to quotations
* Katya - scrap url to get neme of newspaper section, analysis of the dataset
* Wei - results analysis, finding dependencies in the data (in the embedding space)
* Pawel - enrich data from wikipedia


