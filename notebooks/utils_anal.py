#https://stackoverflow.com/questions/63329657/python-3-7-error-unsupported-pickle-protocol-5
#import pickle5 as pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import urllib
#import wget
#rename things that arent logs later
import os
from pathlib import Path
import json
import gzip
from collections import Counter
import datetime
import datasets
from datasets import ClassLabel, Sequence, load_dataset, load_metric
import random
import pandas as pd
from IPython.display import display, HTML
import datetime
import string
import re
from statistics import  median_grouped

def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)



def show_random_elements(dataset, num_examples=3):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    
    for column, typ in dataset.features.items():
        if isinstance(typ, ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
        elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
            df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
    display(HTML(df.to_html()))

class ResultSet(object):
    def __init__(self,saveRoot,dataset, all_train_evals=False):
        """
            Build directory of saved output directories descendent of saveRoot
            that contain N-best predictions files

            arguments: 
                saveRoot (str): a relative or absolyte directory whose descendent
                                directories will be searched for N-best files
                                This distinguishes if these are all evals one training run vs evals of different models.
                dataset (datasets.arrow_dataset.Dataset): A dataset (e.g. validation) which
                            includes question, answer, id, answers for each of the original examples
                all_train_evals (Boolean): N-best files are compressed. They are found in directory
                            nbest and are named: eval_nbest_predictions.999.gz 
                            999 indicates the sequence of the evaluation runs. 
            returns:
                A dictionary keyed on directories that contain eval_nbest_predictions.json files
                with values:
                    mtime (iso datetime): the last modification time of the prediction file
                    config (dict): a dictionary of configuration data used for the run
                    all_results (dict): the evaluation and training results of the run
        """  
        super().__init__() 
        self.all_train_evals = all_train_evals
        self.dataset = dataset
        self.ix_id = dict([(x['id'],i) for i,x in enumerate(dataset)])
        self.saveRoot = Path(saveRoot)
        if all_train_evals:
            nbest_list = list(self.saveRoot.glob('**/eval_nbest_predictions.*.gz'))
        else:
            nbest_list = list(self.saveRoot.glob('**/eval_nbest_predictions.json'))
        self.nbest_list = sorted(nbest_list)
        self.experiments = []
        self.mtimes = []
        self.null_odds = []
        self._topNPreds = []
        self.configs = []
        self.results = []

        for p in self.nbest_list:
            config={}
            all_results = {}
            null_odds =  {}
            pred = {}

            pp = p.parent

            c = list(pp.glob('config.json'))
            if c and not all_train_evals: 
                with open(c[0]) as d:
                    config = json.load(d)

            r = list(pp.glob('all_results.json'))
            if r and not all_train_evals: 
                with open(r[0]) as d:
                    all_results = json.load(d)

            o = list(pp.glob("eval_null_odds.json"))
            if o and not all_train_evals: 
                with open(o[0]) as d:
                    null_odds = json.load(d)

            if (r and o) or all_train_evals:
                if all_train_evals:
                    with gzip.open(p, 'rt', encoding='UTF-8') as zipfile:
                        pred = json.load(zipfile)
                    nm = 'eval'+p.suffixes[0]
                else:
                    with open(p) as d:
                        pred = json.load(d)
                    nm = str(Path(p).parents[0]).replace(saveRoot,'')
                mtime = datetime.datetime.fromtimestamp(p.stat().st_mtime)
                self.experiments.append(nm)
                self.mtimes.append(mtime.isoformat())
                self.configs.append(config)
                self.results.append(all_results)
                self.null_odds.append(null_odds)
                self._topNPreds.append(pred)

 
        for nm, E in zip(self.summaryDF()['experiment'],self._topNPreds):
            for k,v in E.items():
                cxtLen = len(self.dataset[self.ix_id[k]]['context'])
                gL = self.dataset[self.ix_id[k]]['answers']['text']
                ga= list(set(gL))
                for i,x in enumerate(sorted(v,reverse=True,key=lambda s: s['probability'])):
                    x['experiment'] = nm
                    x["id"] = k
                    x["rank"] = i
                    x["goldAns"] = ga
                    if ga == [] :
                        corr = compute_em('',x['text'])==1
                    else:
                        corr = any([compute_em(a,x['text'])==1 for a in ga])
                    x["correct"] = corr
                    x["contextLen"] = cxtLen
        self.labels = self.experiments
    
    def set_labels(self,labels):
        assert len(labels) == len(self.experiments), "Labels list must be the same size as the experiments list"
        self.labels=labels
        
    def summaryDF(self):
        c = [len(x) for x in self._topNPreds]
        k = [ max([len(L) - 1 for L in E.values()]) for E in self._topNPreds ]
        res = pd.DataFrame([self.experiments,self.mtimes,c,k]).transpose()
        res.columns = ['experiment','mtime','example_count', 'topK']
    
        return res

    def topNPreds(self,topK=20, experimentIdx = None):
        if experimentIdx:
            Es = [self._topNPreds[experimentIdx]]
        else:
            Es = self._topNPreds
        results = []
        for E in Es:
            for L in E.values():
                for x in L:
                    x['bin'] = min(x['rank'],topK)
            results.append(E)
        return results

    def topNPredsDF(self,topK=20, experimentIdx = None):
        if experimentIdx:
            Es = [self._topNPreds[experimentIdx]]
        else:
            Es = self._topNPreds
        results = []
        for E in Es:
            for L in E.values():
                for x in L:
                    x['bin'] = min(x['rank'],topK)
                    results.append(x)
        return pd.DataFrame(results)

    def show_random_qa_predictions(self, experimentIdx, num_examples=3, verbose = True):
        dataDict = self._topNPreds[experimentIdx]
        assert num_examples <= len(dataDict), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = (random.randint(0, len(dataDict)-1))
            while pick in picks:
                pick = random.randint(0, len(dataDict)-1)
            picks.append(pick)

        keys = [list(dataDict.keys())[i] for i in picks]
        dv = [dataDict[pick] for pick in keys]
        df = pd.DataFrame([r for v in dv for r in v ])
        for column, typ in df.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
            elif isinstance(typ, Sequence) and isinstance(typ.feature, ClassLabel):
                df[column] = df[column].transform(lambda x: [typ.feature.names[i] for i in x])
        df = df.loc[:,['id','rank','text','goldAns','probability','correct']]
        if verbose:
            display(HTML(df.to_html()))
        return(df)

    def calc_golden_ranks(self, maxBins=9999,experimentIdx=None):
        '''
        This method calculates and returns four numpy arrays, each containing a metric for the highest ranking 
        prediction that exactly matches one of the golden answers. The four metrics are: 
        1. The rank of the prediction up to the parameter maxBins. Lower ranks (higher rank numbers) are given the
           same value equal to maxBins. 
        2. The probability of the highest ranking match, or zero if there is no match in the top-K predictions.
        3. The top probability (at rank=0) for the example.
        4. The actual value of maxBins (highest value in the first array)
        '''
        grls = []
        nas = []
        grls_na = []
        grlProbs = []
        topProbs = []
        mediansNonZeroRanks = []
        for E in self.topNPreds(topK=maxBins,experimentIdx=experimentIdx):
            grl = []
            na = []
            #grlProb = []
            #topProb = []
            for id,v in E.items():
                gr = maxBins
                grp = 0
                pr  = v[0]['probability']
                for x in sorted(v,reverse=True,key=lambda s: s['probability']):
                    if x['correct'] == True:
                        gr = x['bin']
                        grp = x['probability']
                        break
                grl.append(gr)
                na.append(v[0]['goldAns']==[])
                #grlProb.append(grp)
                #topProb.append(pr)
            grls.append(grl)
            nas.append(na)
            #grlProbs.append(grlProb)
            #topProbs.append(topProb)
            mediansNonZeroRanks.append(median_grouped([r  for r in grl if r>0]))
        grls_np = np.array(grls)
        nas_np = np.array(nas)
        #grlProbs_np = np.array(grlProbs)
        #topProbs_np = np.array(topProbs)
        mediansNonZeroRanks_np = np.array(mediansNonZeroRanks)
        check = np.array(self.summaryDF()['example_count'])[0]
        #print('grls_np shape:',grls_np.shape)
        assert grls_np.shape[1] == check, 'result length should be equal to keys in pred'
        #assert grlProbs_np.shape[1] == check, 'result length should be equal to keys in pred'
        #assert topProbs_np.shape[1] == check, 'result length should be equal to keys in pred'
        maxBinsActual = np.max(grls_np)
        #print('maxBins:',maxBins)
        # No grlProbs_np, topProbs_np
        return grls_np, nas_np, maxBinsActual, mediansNonZeroRanks_np, grls

    def eval_metrics(self):
        res = [{'EM_ans':E['eval_HasAns_exact'],
                'F1_ans':E['eval_HasAns_f1'],
                'F1-EM_ans_delta':E['eval_HasAns_f1']-E['eval_HasAns_exact'],
                'noans':E['eval_NoAns_exact'],
                'EM':E['eval_exact'],
                'F1':E['eval_f1'],
                'F1-EM_delta':E['eval_f1']-E['eval_exact']} for E in self.results]
        return pd.DataFrame(res)

def plot_golden_EM_by_rank(grls_np, medians=[], cumulative = False,
                        skipRank0=False,labels=None,savePath=None, subject = '' ):
    lowx = 1 if skipRank0 else 0
    title = '%sCounts by GR - %s' % ('Cumulative ' if cumulative else '',subject)
    title = '%s%s' % (title,'\n(Ranks > 0 only)' if skipRank0 else '')
    colors = cm.Paired(np.linspace(0, 1, len(grls_np)))
    m = grls_np.shape[1]
    maxBins = np.max(grls_np)
    bins = [x for x in range(maxBins+2)]
    xticks = [(bins[idx] + value)/2 for idx, value in enumerate(bins[:-1])]     
    xticks_labels = [ "{:.0f}".format(value, bins[idx+1]) for idx, value in enumerate(bins[:-1])]
    xticks_labels[-1] = xticks_labels[-1]+'+'
    plt.xticks(xticks, labels = xticks_labels)
    plt.tick_params(axis='x', which='major',length=0)
    A = plt.hist(grls_np.T,bins=maxBins+1,
            weights=np.ones(grls_np.T.shape)*100/m, 
            cumulative=cumulative,align='left',histtype='bar',
            range=(lowx,(maxBins+1)),
             color=colors)
    if labels:
        if savePath:
            if cumulative:
                plt.legend(labels=labels,loc='lower right', bbox_to_anchor=(1, 0))
            else:
                plt.legend(labels=labels,loc='upper right', bbox_to_anchor=(1, 1))
        else:
            plt.legend(labels=labels,loc='upper left', bbox_to_anchor=(1, 1))
    if len(medians) > 0:
        plt.vlines(x=medians,ymin=0,ymax=100, linestyles=':',color=colors,alpha=0.5)
        title = '%s%s' % (title,'\nDotted Lines show GRIM for ranks > 0')
    plt.title(title)
    # x ticks
    if savePath:
        plt.savefig(savePath,dpi=300,alpha=0.6)
    plt.show()

    return A[0], medians

# Birch clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot
from matplotlib.patches import Rectangle

def cluster_examples_by_GR(X = None, X_means= None, X_std = None, maxBins = 10, K_clusters = 4, slice_descr="All Examples", save_path = None ):
    '''
    Clusters examples by their Golden Rank of each experiment they participate in. It also prints a graph with 
    
    Arguments:
        X = numpy array of shape (NumberOfExamples X NumberOfExperiments) with the GR values
        X_means = Vector of size (NumberOfExamples) holding the mean of the GRs across the experiments
        X_std   = Vector of size (NumberOfExamples) holding the respective standard deviations
        maxBins = The number of GRs from 0 to maxBins -1. An extra bin with value maxBins is used to capture GR >= maxBins
        K_clusters = The number of clusters
        slice_descr = a description of the population being analyzed used in the title of the graph (e.g. "All Examples" or "Unanswerable Examples")
        save = a valid path where to save the plot, if provided.
    Returns:
        A pandas data frame that contains K-cluester + 2 rows. The first row reports the number of examples successfully predicted (GR = 0) all experiments.
        The last row reports the examples that have no experiment produce a GR < maxBins. These are all the examples that have higher GRs for all experiments.
        The remaining rows (one per cluster) are organized in ascending order of their X-axis value, i.e. the first cluster is closer to rank 0. 
        The following columns are reported:

            Characterization:   The first and last row have values "All Correct" and "All Wrong" respectively. The clusters closest to these two ends 
                                contain values "Mostly Correct" and "Mostly Wrong", while the rest carry the value "Uncertain"  
            Counts:             The count of examples in the cluster. The first and last cluster have the "All Correct"  and "All Wrong" values subtracted from
                                the respective clusters they belong, since they are reported in the first and last row.
            formMeans, toMeans  The range of mean values of the cluster (x-axis)
            fromStd, toStd      The ranfe of standard deviations (y-axis)
    '''
    # Cluster examples by their GR in each experiment they participated
    model = Birch(threshold=0.001,n_clusters=K_clusters)
    model.fit(X)
    yhat = model.predict(X) # vector: cluster ID for each example

    clusters = unique(yhat) # unique clusters
    colors = cm.rainbow(np.linspace(0, 1, len(clusters)))

    X0=np.sum(np.sum(X,axis = 1)==0)
    X10=np.sum(np.mean(X,axis = 1)==maxBins)
    df0=pd.DataFrame({'Characterization': 'All Correct','Counts':X0,'fromMeans':0,'toMeans':0,'fromStd':0,'toStd':0},index=[0])
    df10=pd.DataFrame({'Characterization':'All Wrong','Counts':X10,'fromMeans':0,'toMeans':0,'fromStd':0,'toStd':0},index=[0])



    counts_pd = pd.DataFrame([sum(yhat==c) for c in clusters],clusters, columns = ['Counts'])
    toMeans = [max([X_means[i] for i in [x for x in where(yhat==c)][0]]) for c in clusters]
    fromMeans= [min([X_means[i] for i in [x for x in where(yhat==c)][0]]) for c in clusters]
    toStd = [max([X_std[i] for i in [x for x in where(yhat==c)][0]]) for c in clusters]
    fromStd= [min([X_std[i] for i in [x for x in where(yhat==c)][0]]) for c in clusters]
    counts_pd['Characterization'] = 'Uncertain'
    counts_pd['fromMeans'] = fromMeans
    counts_pd['toMeans'] = toMeans
    counts_pd['fromStd'] = fromStd
    counts_pd['toStd'] = toStd
    counts_pd=counts_pd.sort_values('fromMeans')
    counts_pd.loc[counts_pd['fromMeans']==0,'Counts']=counts_pd['Counts']-X0
    counts_pd.loc[counts_pd['toMeans']==maxBins,'Counts']=counts_pd['Counts']-X10
    counts_pd.loc[counts_pd['fromMeans']==0,'Characterization']='Mostly Correct'
    counts_pd.loc[counts_pd['toMeans']==maxBins,'Characterization']='Mostly Wrong'
    counts_pd = pd.concat([df0,counts_pd,df10])
    counts_pd.index=range(len(counts_pd))


    fig, ax = plt.subplots()
    #fig.set_ppi(150.0)
    ax.scatter(X_means,X_std,alpha = 0.3, marker = '.',s=3,color = colors[yhat])

    xys = [(x,y) for x,y in zip(fromMeans,fromStd)]
    ws = [x2-x1 for x1,x2 in zip(fromMeans,toMeans)]
    hs = [y2-y1 for y1,y2 in zip(fromStd,toStd)]
    for xy,w,h,col in zip(xys,ws,hs,colors[clusters]):
        ax.add_patch( Rectangle(xy,w,h,fc='none',
                            color =col,
                            linewidth = 1,
                            linestyle="dashed") )
    plt.xlabel("Example Golden Rank Mean across Experiments")
    plt.ylabel("Standard Deviation")
    plt.title("Examples Clustered by GR of each Experiment\nBirch Clustering - K=%i - %s"%(K_clusters,slice_descr))
    if save_path:
        plt.savefig(str(save_path),dpi=300.0)
    fig.show()

    return counts_pd