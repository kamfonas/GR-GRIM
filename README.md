# Golden Rank (GR) and Golden Rank Interpolated Median (GRIM) 

This repository accompanies the article *What can Secondary Predictions Tell us?: An Exploration on Question-Answering with SQuAD-v2.0* by Michael Kamfonas and Gabriel Alon which can be found [here](https://arxiv.org/abs/2206.14348).

Briefly, the Golden Rank is the rank of the best prediction, in predictive probability order, that matches a golden truth answer. For successful predictions, this rank is 0, for failed (secondary)  predictions it is a higher rank, whose magnitude may be attributed either to the example (difficulty, flaw) or by the model (architecture, training). We apply our analysis on results of the question-answering task on the SQuAD-v2.0 dataset. 


## In order to run the notebooks you will need the following:

1.  Clone this repository into a local directory (we will refer to it as the base directory)
1.  Access to the SQuAD-v2.0 dataset. This is done for you in each notebook, in the section titled **Load the SQuAD Dataset**, using the `load_dataset` method from package `datasets`. After the dataset is downloaded, data gets cached automatically and can be reloaded nearly instantaneously for every subsequent execution of the command.

2.  Download the [save zipped folder](https://drive.google.com/file/d/1z4M-JJhBSueK8ncNfb1RaRpv4gFgmA6Q/view?usp=sharing) and the [save-epochs zipped folder](https://drive.google.com/file/d/1Fpeu0J7XoCpwYtFUrdMlzZVGhrGwk5EI/view?usp=sharing), unzip the files and make sure the save and save-epochs folders are in the base directory.
The output from each experiment resides in a subdirectory of one of these two folders: (a) Experiment folders intended for model comparitive analysis are formed as `store/squad_v2/<model-name>/999` where model name is the Hugging Face transformer name (e.g. `navteca/electra-base-squad2`) and 999 stands for a three-digit sequence number that distinguishes multiple experiments using the same model. (b) For comparing all validations during a single fine-tuning training run, experiment folders are formed as `save-epochs/squad_v2/<model-name>/999` and contain a subfolder called `evals` which holds a separate zip file for each evaluation executed during the run. 
5.  The notebooks analyze data collected from evaluation runs of the `run_qa` application of the Hugging Face `transformers` library. This data include the file `eval_nbest_predictions.json` which stores the top N predictions of each example evaluated, with its logits, probability and predicted answer text. By default, the program `run_qa`, when executed in training mode, produces the 20-best predictions per example for each evaluation, so that new evaluations override the same file name with the latest one surviving before the run ends or gets interrupted. Our experiments used the argument `n_best_size=10` limiting the output to only the 10-best predictions to economize on time and space. 
6. In order to collect all evaluations of a training run we modified the program to zip and save each evaluation file in a separate zip-file inside the `eval` directory as mentioned in section 2. Notice that although we use the name `save-epochs`, the evaluations were taken after a constant number of optimizer training iterations, and may not coincide with the end of an epoch.

## The notebooks included are:

<dl><dt>Analysis_using_hf_logs_v5_mk</dt>
  <dd>This is the analysis derived from fine-tuning results, based on the final evaluation, of multiple models and runs stored in separate experiment folders inside the save/squad_v2 directory. You can follow the same naming convention and add your own runs to the same or other directory, and adjust the first parameter of the box under <em>Create a Class Instance and List Summary Info</em>. The ResultSet class intialization will search all subfolders and pick-up the required information from any subfolders that contain the required json output files: <b>all_results</b>, <b>eval_null_odds</b> and <b>eval_nbest_predictions</b> </dd>
<dt>Analysis_using_hf_logs_TrainingSnapshots_mk</dt>
  <dd>This is the analysis based on all validations within a multi-epoch single fine-tuning run of a <b>RoBERTa-base model</b></dd>
<dt>Analysis_using_hf_logs_TrainingSnapshots2_mk</dt>
  <dd>This is the analysis of all validations captured within additional fine tuning runs on a <b>BERT-large-uncased whole-word-masking</b> model by Deepset, already pretrained on SQuAD-v2.0</dd>
</dl>

## Reference

If you use our work, please cite our paper:

@misc{https://doi.org/10.48550/arxiv.2206.14348,
  doi = {10.48550/ARXIV.2206.14348},  
  url = {https://arxiv.org/abs/2206.14348},
  author = {Kamfonas, Michael and Alon, Gabriel},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {What Can Secondary Predictions Tell Us? An Exploration on Question-Answering with SQuAD-v2.0},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
