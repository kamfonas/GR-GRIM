# Golden Rank (GR) and Golden Rank Interpolated Median (GRIM) 

This repository accompanies the article *What can Secondary Predictions Tell us?: An Exploration on Question-Answering with SQuAD-v2.0* by Michael Kamfonas and Gabriel Alon which can be found [here](https://arxiv.org/abs/2206.14348).

Briefly, the Golden Rank is the rank of the best prediction, in predictive probability order, that matches a golden truth answer. For successful predictions, this rank is 0, for failed predictions it is a higher rank, that can be attributed either to the example (difficulty, flaw) or by the model (architecture, training). We apply our analysis on results of the question-answering task on the SQuAD-v2.0 dataset. 


In order to run the notebooks you will need the following:

1.  Access to the SQuAD-v2.0 dataset. This is done for you in each notebook, in the section titled **Load the SQuAD Dataset**, using the `load_dataset` method from package `datasets`. After the dataset is downloaded, data gets cached automatically and can be reloaded nearly instantaneously for every subsequent execution of the command.

1.  The notebooks analyze data collected from evaluation runs of the `run_qa` application of the Hugging Face `transformers` library. This data include the file `eval_nbest_predictions.json` which stores the top N predictions of each example, including logits, probability and answer. In its native form, program `run_qa` produces 20-best predictions per example from each evaluation executed during training, so that new evaluations override the previous file leaving the last one after the run ends. We used the argument `n_best_size=10` limiting the output to the 10-best predictions, and saved the output of each run in directory `store/squad_v2/<model-name>/999` where model name is the Hugging Face transformer name (e.g. `navteca/electra-base-squad2`) and 999 stands for a three-digit sequence number distinguishing multiple experiments when using the same model.
2. In order to collect all evaluations of a training run we modified the program to zip and save each evaluation file in a separate zip-file inside an `eval` directory in directory `save-epochs/squad_v2/<model-name>/999`. Notice that although we use the name `save-epochs`, the evaluations were taken after a constant number of optimizer training iterations, and may not coincide with the end of an epoch.
