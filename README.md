# Golden Rank (GR) and Golden Rank Interpolated Median (GRIM) 

This repository accompanies the article *What can Secondary Predictions Tell us? An Exploration on Question-Answering with SQuAD-v2.0* by Michael Kamfonas and Gabriel Alon which can be found [here](https://arxiv.org/abs/2206.14348).

Briefly, the Golden Rank is the rank of the best prediction, in predictive probability order, that matches a golden truth answer. For successful predictions this rank is 0. For failed (secondary)  predictions it is a higher integer, whose magnitude may be attributed either to the example (e.g. difficulty, flawed formulation) or by the model (e.g. architecture limitations, inadequate training). We apply our analysis on results of the question-answering task on the SQuAD-v2.0 dataset. 


## In order to run the notebooks you will need the following:

1.  Clone this repository into a local directory (we will refer to it as the base directory)
1.  Access to the SQuAD-v2.0 dataset: This is done for you in each notebook, in the section titled **Load the SQuAD Dataset**, using the `load_dataset` method from package `datasets`. After the dataset is downloaded, data gets cached automatically and can be reloaded nearly instantaneously for every subsequent execution of the command.

2.  Download the [save-models zipped folder](https://drive.google.com/file/d/1GytKwK_kqjbfQAfgVcMeOFkF_l2jpL4c/view?usp=sharing) and the [save-epochs zipped folder](https://drive.google.com/file/d/1Fpeu0J7XoCpwYtFUrdMlzZVGhrGwk5EI/view?usp=sharing), unzip the files and make sure the save-models and save-epochs folders are in the base directory. 
The output from each of our experiments resides in a subdirectory of one of these two folders: (a) Experiment folders intended for model comparative analysis are formed as `store-models/squad_v2/<model-name>/999` where model name is the Hugging Face transformer name (e.g. `navteca/electra-base-squad2`) and 999 stands for a three-digit sequence number that distinguishes multiple experiments using the same model. (b) For comparing all validations during a single fine-tuning training run, experiment folders are formed as `save-epochs/squad_v2/<model-name>/999` and contain a subfolder called `evals` which holds a separate zip file for each evaluation executed during the run. A third folder, named `save` can be used to store and run model analysis on your own evaluation output as explained in the last section of this page.
5.  The notebooks analyze data collected from evaluation runs of the `run_qa` application of the Hugging Face `transformers` library. This data include the file `eval_nbest_predictions.json` which stores the top N predictions of each example evaluated, with its logits, probability and predicted answer text. By default, the program `run_qa`, when executed in training mode, produces the 20-best predictions per example for each evaluation, so that each new evaluation overrides the same file, with the latest one surviving when the run ends or gets interrupted. Our experiments used the argument `n_best_size=10` limiting the output to only the 10-best predictions to economize on time and space. 
6. In order to collect all evaluations of a training run we modified the program to save each validation's output `eval_nbest_predictions.json` as a separate zip-file in a subdirectory of the experiment folder called `eval`, as mentioned in section 3. Notice that although we use the name `save-epochs`, the evaluations were taken after a constant number of optimizer training iterations, and may not coincide with the end of an epoch.

## The notebooks included are:

<dl><dt>Analysis_using_hf_logs_v5_mk</dt>
  <dd>This is the analysis derived from fine-tuning results of the final evaluation of each experiment. It is intended to compare different models and experiments based on their final evaluation. Each run is stored in separate experiment folders inside the save-models/squad_v2 directory. We recommend that you use a third directory, named "save/squad_v2" and follow the same naming convention to add your own runs for analysis. You will have to adjust the first parameter of the command  under <em>Create a Class Instance and List Summary Info</em>. The ResultSet class initialization will search all subfolders and pick-up the required information from any subfolders that contain the required json output files: <b>all_results</b>, <b>eval_null_odds</b> and <b>eval_nbest_predictions</b>. We offer some scaffolding at the end of this page to help you get started. </dd>
<dt>Analysis_using_hf_logs_TrainingSnapshots_mk</dt>
  <dd>This is the analysis based on all validations within a multi-epoch single fine-tuning run of a <b>RoBERTa-base model</b></dd>
<dt>Analysis_using_hf_logs_TrainingSnapshots2_mk</dt>
  <dd>This is the analysis of all validations captured within additional fine tuning runs on a <b>BERT-large-uncased whole-word-masking</b> model by Deepset, already pretrained on SQuAD-v2.0</dd>
</dl>

## To try our analysis on your own experiments:

We include the `run_qa.py`, `trainer_qa.py` and `utils_qa.py` files from the Hugging Face QA examples of the transformers package version 4.21.0.dev0 that can be found [here](https://github.com/huggingface/transformers/tree/main/examples/pytorch/question-answering). You can opt to install the current or any other version, but ensure that these three python files are replaced with those compatible with the `transformers` package version you choose to install. Assuming you want to use  version 4.21.0.dev0, here is what you have to do:

1.  Create a virtual environment either through conda or in python and activate it.
1.  Install the transformer package release 4.21.0.dev0 or later from source using [these](https://huggingface.co/docs/transformers/installation#installing-from-source) instructions. 
2.  Install the torch/cuda combination compatible with your GPU, and add numpy, pandas, seaborn,scikit-learn, matplotlib, datasets. If additional packages are missing, install them in the environment.  
2.  Create the nested directories `save/squad_v2` at the base of your project folder
3.  We have provided a unix script to run a few validations using different models. Open a terminal and change to the base directory of your project. Run the provided shell script `run_model_evals.sh` (you may have to change its permissions first)
4.  Make a copy of the notebook `Analysis_using_hf_logs_v5_mk.ipynb` inside the notebooks folder and change the command in the cell under the heading *Create  Class Instance and List Summary Info*   to look like this: `RS = ResultSet("../save/squad_v2",raw_datasets["validation"])` and run all cells.

## Reference

If you use our work, please cite our paper:

```
  @misc{https://doi.org/10.48550/arxiv.2206.14348,
  doi = {10.48550/ARXIV.2206.14348},  
  url = {https://arxiv.org/abs/2206.14348},
  author = {Kamfonas, Michael and Alon, Gabriel},
  keywords = {Computation and Language (cs.CL), Artificial Intelligence (cs.AI), Machine Learning (cs.LG), 
              FOS: Computer and information sciences},
  title = {What Can Secondary Predictions Tell Us? An Exploration on Question-Answering with SQuAD-v2.0},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
