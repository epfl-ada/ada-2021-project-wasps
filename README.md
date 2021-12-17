# Detect profile of the speaker based on deep learning
                                            
## Abstract &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [<img src='figures/logo2.png'>](https://weishi7.github.io/)
While performing analysis of Quotebank data we found out that around 34% of quotations don't have assigned speakers to it (1.8 million out of 5.2 million in file quotes-2020.json). Our goal is to answer the following question: if we cannot determine the exact author of a quotation, what other information can we get from it?

In that work we would like to extract additional information about known authors parsing information from Wikipedia and use it to describe unknown authors thereby reducing their obscurity. With the data and labels, we trained several models and verified the functionalities, then predicted the features of the quotations that are not assigned speakers in Quotebank. Also, we did some analysis on the outcomes and explored the relationships between different features, as well as tried to understand the mechanism of the prediction.

Click to the WASPS icon or [here](https://weishi7.github.io/) to check our datastory.

## Repository structure

```notebooks/analysis```: Contains notebooks with data analysis. We have built a graph of the distribution of classes for the analyzed data. In addition, an analysis of the dependencies between classes and quote topics has been added to this notebook.

```notebooks/parsing```: Contains notebooks for the data parsing and splitting data into train, validation and test sets.

```notebooks/snippets```: Contains notebook with explanations how we worked with Parquet data type.

```scripts```: Contain python file with our classification model.

```data```: Contains  data parsed from Wikipedia. The analyzed data is divided into several files depending on the investigated feature.

```figures```: Contains plots that we are using for the README.

## Contribution

_Pawel Mlyniec_: collection data from wikipedia dumps, parsing all the data, sptilling the data into feature categoties, adapting data format for the usage.

_Sofia Blinova_: creation classification model, writing training, validation and prediction piplines, result analyzation.

_Ekaterina Trimbach_: creation most of the visualization, predicted labels and topic analization.

_Wei Shi_: Result analysis on preliminary experiment, Website configuration, models training and records, write huge part of the report.

_All together_: work on the final report.

