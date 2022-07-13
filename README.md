# Golden Rank (GR) and Golden Rank Interpolated Median (GRIM) 

This repository accompanies the article *What can Secondary Predictions Tell us?: An Exploration on Question-Answering with SQuAD-v2.0* by Michael Kamfonas and Gabriel Alon which can be found [here](https://arxiv.org/abs/2206.14348).

Briefly, the Golden Rank is the rank of the best prediction in predictive probability order that matches a golden truth answer. For successful predictions, this rank is 0, for failed predictions it is a higher rank. We apply our analysis on results of the question-answering task on the SQuAD-v2.0 dataset. 


In order to run the notebooks you will need the following:

1.  Access to the SQuAD-v2.0 dataset. This is done in the section titled **Load the SQuAD Dataset** in each of the notebooks, using the `load_dataset` method from package `datasets`. After the dataset is downloaded data get cached automatically and can be reloaded nearly instantaneously for every subsequent execution of the command.

1.  The notebooks use data collected from evaluation runs of the `run_qa` application of the Hugging Face `transformers` library This data include the file `eval_nbest_predictions.json` which provides data from the last
