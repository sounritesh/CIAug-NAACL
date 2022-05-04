!pip install textacy==0.6.3
!pip install pytorch_pretrained_bert

import csv
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
import pandas as pd
import math
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import torch.multiprocessing as mp
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, confusion_matrix, accuracy_score
import numpy as np
import os
from random import shuffle
from datetime import datetime
import time
import logging
import copy
import random
from torch.multiprocessing import Pool
torch.multiprocessing.set_sharing_strategy('file_system')
from functools import partial

import csv
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import pickle
import pandas as pd
import math
import nltk
nltk.download('punkt')
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import classification_report
import bisect
import os
import pdb
import logging
import sys
import socket
import re
import time
from scipy.stats import spearmanr
USE_COLAB = False

logger = logging.getLogger('')

import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])|(\d+)")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
NO_SPACE = ""
SPACE = " "

s_perc_dict = {
    0: 0.1,
    1: 0.2,
    2: 0.3,
    3: 0.4,
    4: 0.5,
    5: 0.6,
    6: 0.7,
    7: 0.8,
    8: 0.9,
    9: 1,

}

def preprocess_reviews(reviews):
    reviews = REPLACE_NO_SPACE.sub(NO_SPACE, reviews.lower())
    reviews = REPLACE_WITH_SPACE.sub(SPACE, reviews) 
    return reviews


def eval(preds, y):
    assert len(preds) == len(y)
    z = np.zeros(len(preds))
    for i, p in enumerate(preds):
      if (p-math.floor(p)) < 0.5:
        z[i] = math.floor(p)
      else:
        z[i] = math.floor(p) + 1
    
    prec_score = precision_score(np.array(y), z,average="micro")
    rec_score = recall_score(np.array(y), z,average="micro")
    f1_score = (2 * prec_score * rec_score) / (prec_score + rec_score)
    # making other metric 0 as they dont signify anything in multiclass
    roc_auc, tn, fp, fn, tp, error_rate = 0,0,0,0,0,0
    return (prec_score, rec_score, f1_score, roc_auc, tn, fp, fn, tp, error_rate)
  
def Average(lst): 
    return sum(lst) / len(lst) 

def trainMix(model, scheduler, optimizer, numEpochs, train_dataloader, eval_dataloader, outLocation, out_file, device, n_gpu = 1 ):
    
    performance = []
    error_rates = [] 
    print("N_GPU", n_gpu)
    # training
    logger.info("Number of Epochs: {}".format(numEpochs))
    for epoch in range(numEpochs):
        train_loss = 0
        correct = 0
        total = 0
        train_preds = []
        train_targets = []
        performance.append({})
        # looping through the training set
        if epoch<10:
          s_perc = s_perc_dict[epoch]
        else:
          s_perc = 1
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, ids = batch

            loss, logits, lam = model(input_ids, segment_ids, input_mask, label_ids, ids,1,s_perc)

            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            loss.backward()
           
            total += label_ids.size(0)
            train_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            correct += logits.eq(label_ids).cpu().sum().float()

        logger.info('epoch:' + str(epoch) + ' loss:' + str(train_loss) + ' Accuracy:' + str(correct/total))

        del train_preds
        del train_loss
        del train_targets

        if eval_dataloader is not None:
            model.eval()
            test_preds = []
            test_targets = []
            for input_ids, input_mask, segment_ids, label_ids, ids in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    logits = model(input_ids, segment_ids, input_mask)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    test_preds.append(logits)
                    test_targets.append(label_ids)

            test_preds = [k[i] for k in test_preds for i in range(k.shape[0])]
            test_targets = [i for item in test_targets for i in item]

            preds = np.array(test_preds)
            test_prediction = np.array(test_targets)
            np.save("test_preds.npy",preds)
            np.save("test_targets.npy",test_prediction)

            test_eval = eval(test_preds, test_targets)
            logger.info('epoch:' + str(epoch) + ' precision:' + str(test_eval[0]) + ' recall:' +
                        str(test_eval[1]) + ' f1:' + str(test_eval[2]) + ' roc_auc:' + str(test_eval[3]) + 
                        ' false positive:' + str(test_eval[5]) + ' Error Rate:' + str(test_eval[8]))
            del test_preds
            del test_targets
        logger.info("Model File: {}".format(out_file +  str(epoch) + '.bin'))



def setup_logger(logger_name, log_file, level=logging.INFO):
    '''This sets up a python logger that follows the amazon guidelines for logging.
    '''
    log = logging.getLogger('')
    formatter = logging.Formatter("%(asctime)s crm_logger %(process)d-0@" +
                                  socket.gethostname() +
                                  ":0 [%(levelname)s] %(filename)s:%(lineno)d " +
                                  "%(message)s",
                                  "%c")
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    log.setLevel(level)
    log.addHandler(stream_handler)
    log.addHandler(file_handler)

modelType = 'BERT_MR_MIX-UP_10_2e-5_single no pre attn full hidden layer 00-8 BCELOSS fixed'

dataStorageLocation = './MR'
logFolder = './MR/logs'

args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": dataStorageLocation,
    "data_dir": dataStorageLocation,
    "task_name": "news_cat_label",
    "no_cuda": False,
    "bert_model": 'bert-base-multilingual-uncased',
    "max_seq_length": 61,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 8,
    "learning_rate": 2e-5,
    "num_train_epochs": 13.0,
    "warmup_proportion": 0.1,
    "no_cuda": False,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}
if not os.path.exists(dataStorageLocation):
    os.makedirs(dataStorageLocation)

if not os.path.exists(logFolder):
    os.makedirs(logFolder)

lr = args['learning_rate']
numEpochs = args['num_train_epochs']
batch_size = args['train_batch_size']

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, labels=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            labels: (Optional) [string]. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,ids):
        self.input_ids = input_ids
        self.input_mask = input_mask  # attention_mask
        self.segment_ids = segment_ids # token_type_ids
        self.label_ids = label_ids
        self.ids = ids

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError() 

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class LabelTextProcessor(DataProcessor):
    
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labels = None
    
    
    def get_train_examples(self, data, size=-1):
        #filename = 'train.csv'
        #logger.info("LOOKING AT {}".format(os.path.join(data_dir, filename)))
        if size == -1:
            #data_df = os.path.join(data_dir, filename),engine=None
            return self._create_examples(data, "train")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            return self._create_examples(data_df.sample(size), "train")
        
    def get_dev_examples(self, dev, size=-1):
        """See base class."""
        #filename = 'test.csv'
        if size == -1:
            #data_df = os.path.join(data_dir, filename)
            return self._create_examples(dev, "dev")
        else:
            data_df = pd.read_csv(os.path.join(data_dir, filename))
            return self._create_examples(data_df.sample(size), "dev")
    
    def get_test_examples(self, data_dir, data_file_name, size=-1):
        data_df = pd.read_csv(os.path.join(data_dir, data_file_name))
        if size == -1:
            return self._create_examples(data_df, "test")
        else:
            return self._create_examples(data_df.sample(size), "test")

    def get_labels(self):
        """See base class."""
        a = [x for x in range(2)]
        return a

    def _create_examples(self, data, set_type, labels_available=True):
        """Creates examples for the training and dev sets."""
        guid = data['text_id']
        text = (data['text'])
        text_a = text
        labels = int(data['label'])
        examples = (InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(label_list, max_seq_length, tokenizer, train_examples):
    """Loads a data file into a list of `InputBatch`s."""
    example = train_examples
    label_map = {label : i for i, label in enumerate(label_list)}
    id = example.guid
    
    tokens_a = tokenizer.tokenize(example.text_a)

    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    if tokens_b:
        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_ids = label_map[example.labels]

    features = (
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          ids = id))
    return features

processors = {
    "news_cat_label": LabelTextProcessor
}

# Setup GPU parameters
if args["local_rank"] == -1 or args["no_cuda"]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args["no_cuda"] else "cpu")
    n_gpu = torch.cuda.device_count()
    n_gpu = 1

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
if n_gpu > 0:
    torch.cuda.manual_seed_all(args['seed'])

task_name = args['task_name'].lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))


filedate = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
setup_logger('DummyLoggerName', os.path.join(logFolder,'CV_'+modelType+'_'+filedate+'_'+'.log'))
logger = logging.getLogger('')

logger.info("Model Type: {}".format(modelType))

import pandas as pd
# give path to the dataset
df = './Load the train dataframe'
df.columns = 'label','Text'
df1 = './Load the test dataframe'
df1.columns = 'label','Text'
print(df.shape)
df = pd.concat([df,df1])

df.shape

index = [x for x in range(len(df))]
index = list(index)

sentences, labels = list(df['Text']), list(df['label'])

l = [len(word.split()) for word in sentences]

len(sentences)

# MR dataset preprocessing 
processor = processors[task_name](args['data_dir'])
label_list = processor.get_labels()
num_labels = len(label_list)

data = []
test_data = [] 
i = 0
for line, label, id in zip(sentences, labels,index):
  # here i< length of training set
    if i < 3250:
      data.append({})
      data[-1]['text_id'] = i
      data[-1]['text'] = line.strip()
      data[-1]['label'] = label
      data[-1]['text_id'] = id
      i +=1
    else:
      test_data.append({})
      test_data[-1]['text_id'] = i
      test_data[-1]['text'] = line.strip()
      test_data[-1]['label'] = label 
      test_data[-1]['text_id'] = id
      i +=1
    

logger.info("--- Pre-processing training data ---")
shuffle(data)
shuffle(test_data)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
train_examples = None
num_train_steps = None
processes = []

pool = Pool(10)
train_examples = pool.map(processor.get_train_examples, data)
eval_examples = pool.map(processor.get_dev_examples, test_data)
pool.close()
pool.join()

pool = Pool(20)
func = partial(convert_examples_to_features, label_list, args['max_seq_length'], tokenizer)
train_features = pool.map(func, train_examples)
pool.close()
pool.join()

pool = Pool(20)
func = partial(convert_examples_to_features, label_list, args['max_seq_length'], tokenizer)
eval_features = pool.map(func, eval_examples)
pool.close()
pool.join()

print(len(train_features))
print(len(eval_features))

print("Features generated --", len(train_features))

logger.info("Training Model")
logger.info("Learning Rate: {}".format(lr))
logger.info("***** Running training *****")
logger.info("  Num examples = %d", len(train_examples))
logger.info("  Batch size = %d", args['train_batch_size'])

all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
all_ids = torch.tensor([f.ids for f in train_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_ids)
train_dataloader = DataLoader(train_data, batch_size=args['train_batch_size'], shuffle=True)

logger.info("***** Building Eval DataLoader *****")
logger.info("  Num examples = %d", len(eval_examples))
logger.info("  Batch size = %d", args['eval_batch_size'])
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_ids = torch.tensor([f.ids for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_ids)
# Run prediction for full data
eval_dataloader = DataLoader(eval_data, batch_size=args['eval_batch_size'])

def get_cosine_sentence(i,perc,common,a,s_perc):
  num = np.amax(a[i])
  p = num*perc
  args = np.argwhere(a[i]>p.item())
  args_vals = []
  for j in args.flatten():
    args_vals.append(a[i][j])

  args_vals = np.sort(np.array(args_vals))
  
  num_samps_in_range = int(round(s_perc*len(args)))
  try:
    if s_perc == 0 or num_samps_in_range == 0:
      rand_val = np.amin(args_vals)
    else:
      vals_in_range = args_vals[:num_samps_in_range]
      r = np.random.randint(0,len(vals_in_range),1)
      rand_val = vals_in_range[r[0]]

    rand = np.where(a[i]==rand_val)[0][0]
    return rand

  except:
    print("Some issue")
    return args[0]

  return a

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

master_ids = np.array([f.ids for f in train_features])

master_input_ids = [f.input_ids for f in train_features]
master_input_mask = [f.input_mask for f in train_features]
master_segment_ids = [f.segment_ids for f in train_features]
master_label_ids = [f.label_ids for f in train_features]

def get_second_example(ids,common,a,s_perc):
  array = []
  idx = []
  # Change the Threshold value as required.
  percentage = 0.80
  for i in range(len(a)):
    a[i] = common[i].item()*a[i]
  for i in range(len(ids)):
    ex = ids[i]
    exs = ex.item()
    sel = get_cosine_sentence(ex,percentage,common,a,s_perc)
    idx.append(sel)
    array.append(common[ids[i]]*(torch.tensor(a[ids[i]][sel].astype(np.float32),device="cuda")))

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for i in range(len(idx)):
    num = idx[i]
    pos = np.argwhere(master_ids == num)
    pos = int(pos)
    all_input_ids.append(master_input_ids[pos])
    all_input_mask.append(master_input_mask[pos])
    all_segment_ids.append(master_segment_ids[pos])
    all_label_ids.append(master_label_ids[pos])

  all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
  all_input_mask = torch.tensor(all_input_mask, dtype=torch.long)
  all_segment_ids = torch.tensor(all_segment_ids, dtype=torch.long)
  all_label_ids = torch.tensor(all_label_ids, dtype=torch.long)
  train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
  train_dataloader = DataLoader(train_data, batch_size=args['train_batch_size'])

  return train_dataloader,array

"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import json
import logging
import os
import six
import shutil
import tempfile
import fnmatch
from functools import wraps
from hashlib import sha256
from io import open

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm

try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pytorch_transformers')

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                        default_cache_path))

PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility

WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

if not six.PY2:
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = ''.join(docstr) + fn.__doc__
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = fn.__doc__ + ''.join(docstr)
            return fn
        return docstring_decorator
else:
    # Not possible to update class docstrings on python2
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator

def url_to_filename(url, etag=None):
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    """
    url_bytes = url.encode('utf-8')
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir=cache_dir, force_download=force_download, proxies=proxies)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file, proxies=None):
    req = requests.get(url, stream=True, proxies=proxies)
    content_length = req.headers.get('Content-Length')
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B", total=total)
    for chunk in req.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None):
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies)
            if response.status_code != 200:
                etag = None
            else:
                etag = response.headers.get("ETag")
        except EnvironmentError:
            etag = None

    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])

    if not os.path.exists(cache_path) or force_download:
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                s3_get(url, temp_file, proxies=proxies)
            else:
                http_get(url, temp_file, proxies=proxies)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')  # The beauty of python 2
                meta_file.write(output_string)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path

import tempfile
import tarfile
import json
import shutil

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}

BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'
class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.
        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12):
        """Constructs BertConfig.
        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

def mixup_hidden_states(x, y, matrix1, alpha=1.0, use_cuda=True):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    x = x.float()

    matrix = torch.tensor(matrix1,device="cuda")
    matrix = matrix.float()
    indices = np.random.permutation(x.size(0))
    mixed_x = torch.zeros_like(x)


    for i in range(len(matrix1)):
      mixed_x[i] = x[i]*matrix1[i] + y[i]*(1-matrix1[i])

    batch_size = x.size()[0]

    return mixed_x, matrix1, indices

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, hidden_states2=None, attention_mask2=None, matrix=None, output_all_encoded_layers=True, mixup=None):
        lam = None
        index = None

        all_encoder_layers = []
        all_encoder_layers2 = []
        layer = 0
        #mixup_layer = random.randint(0,10)
        #mixup_layer = 1
        mixup_layer_set = [6, 7, 9, 12]
        mixup_layer = np.random.choice(mixup_layer_set, 1)[0]
        mixup_layer = mixup_layer - 1
        for layer_module in self.layer:
            if mixup is not None:
                if layer == mixup_layer:
                    hidden_states, lam, index = mixup_hidden_states(hidden_states,hidden_states2, matrix, mixup)
            layer +=1
            hidden_states = layer_module(hidden_states, attention_mask)
            if hidden_states2 is not None:
              hidden_states2 = layer_module(hidden_states2, attention_mask2)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                if hidden_states2 is not None:
                  all_encoder_layers2.append(hidden_states2)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        if mixup is not None:
            return all_encoder_layers, lam, index
        
        return all_encoder_layers
class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
    
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def to_one_hot(inp,num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()

    y_onehot.scatter_(1, inp.unsqueeze(1).data.cpu(), 1)
    
    #return Variable(y_onehot.cuda(),requires_grad=False)
    return y_onehot

def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted

bce_loss = nn.BCELoss().cuda()
softmax = nn.Softmax(dim=1).cuda()
criterion = nn.CrossEntropyLoss().cuda()

def mixup_bertEmbedding(x, alpha=1.0, use_cuda=True):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    x = x.float()
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, lam, index

def mixup_labels(target_reweighted, target_reweighted2, lam):
    #target_shuffled_onehot = target_reweighted
    target_reweighted = target_reweighted * lam + target_reweighted2 * (1 - lam)
#     y_a, y_b = y, y[index]
    return target_reweighted

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    y_a = y_a.float()
    y_b = y_b.float()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").
    Params:
        config: a BertConfig class instance with the configuration to build a new model
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.
    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, input_ids2=None, token_type_ids2=None, matrix=None, attention_mask=None,attention_mask2 =None, output_all_encoded_layers=True, mixup=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask2 is None:
            try:
              attention_mask2 = torch.ones_like(input_ids2)
            except:
              attention_mask2 = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        if token_type_ids2 is None:
            try:
              token_type_ids2 = torch.zeros_like(input_ids2)
            except:
              token_type_ids2 = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask2 = attention_mask2.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask2 = extended_attention_mask2.to(dtype=next(self.parameters()).dtype) # fp16 compatibility

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask2 = (1.0 - extended_attention_mask2) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)  #Shape - 32, 250, 768
        try:
          embedding_output2 = self.embeddings(input_ids2, token_type_ids2)  #Shape - 32, 250, 768
        except:
          embedding_output2 = self.embeddings(input_ids, token_type_ids)
        
#         if mixup is not None:
#             embedding_output, lam, index = mixup_bertEmbedding(embedding_output, mixup)
        if mixup is not None:
            encoded_layers, lam, index = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          embedding_output2,
                                          extended_attention_mask2,
                                          matrix,
                                          output_all_encoded_layers=output_all_encoded_layers,
                                          mixup=mixup)
        else:
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        
        if mixup is not None:
            lam_tensor = torch.zeros(pooled_output.size(0),1).cuda()
            lam_tensor = lam
            return encoded_layers, pooled_output, lam_tensor, index
        
        return encoded_layers, pooled_output
    
class BertForSequenceClassificationMix(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    num_labels = 2
    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(BertForSequenceClassificationMix, self).__init__(config)
        self.num_labels = 2
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.apply(self.init_bert_weights)
        # Replace with path to the calculated Hyperbolic Distance Matrix
        self.a = np.load('./Load the matrix')
        self.a = self.a/np.max(self.a)

        self.common = [nn.Parameter(torch.tensor([1.],device="cuda"),requires_grad=True) for x in range(len(self.a))]

        self.criterion = nn.BCELoss(reduce=True)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,ids = None, alpha=None, s_perc=1):
        lam = None
        if(ids is not None):
          ids = ids.to("cpu")
          second_dataloader,matrix = get_second_example(ids,self.common,self.a, s_perc)
          for step1, batch1 in enumerate(second_dataloader):
            batch1 = tuple(t.to(device) for t in batch1)
            input_ids2, attention_mask2, token_type_ids2, labels2 = batch1
        
        if labels is not None:
            _, pooled_output, lam, index = self.bert(input_ids, token_type_ids, input_ids2, token_type_ids2, matrix, None, output_all_encoded_layers=False,
                                                     mixup=1)
            target_reweighted = to_one_hot(labels, num_labels)
            target_reweighted2 = to_one_hot(labels2, num_labels)

            mixed_target = mixup_labels(target_reweighted, target_reweighted2, lam[0].item())
            mixed_target = mixed_target.cuda()
        else: 
            _, pooled_output = self.bert(input_ids, token_type_ids, None, None, None, attention_mask, None, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = (self.classifier(pooled_output))
        values, indices =torch.max(logits, 1)

        if labels is not None:
            labels = labels.float()
            loss = bce_loss(softmax(logits), mixed_target)
            return loss , indices, lam[0].item()
        else:
            return indices
        
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def freeze_bert_embedding(self):
        for name,param in model.named_parameters():
            if name.startswith('bert.embeddings'):
                param.require_grad = False
                print(name)
    
    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def set_multiple_gpus(self):
        # here uses multi gpu
        self.bert = nn.DataParallel(self.bert, device_ids=[0,1,2,3])

torch.cuda.set_device(0)
print("Train examples --", len(train_examples))
logger.info("Initializing Model")
model = BertForSequenceClassificationMix.from_pretrained(args['bert_model'], num_labels = num_labels)
#model.freeze_bert_embedding()
if n_gpu > 1:
    model.set_multiple_gpus()
model.to(device)

param_optimizer =list(model.named_parameters())
for i in range(len(model.common)):
    param_optimizer.append((f'number{i}',model.common[i]))

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
num_train_steps = len(train_dataloader) / args['gradient_accumulation_steps'] * args['num_train_epochs']
t_total = num_train_steps
optimizer = BertAdam(optimizer_grouped_parameters,
                     lr=args['learning_rate'],
                     warmup=args['warmup_proportion'],
                     t_total=t_total)
logger.info("Initializing Model -- DONE")

trainMix(model, None,  optimizer, int(numEpochs), train_dataloader, eval_dataloader, dataStorageLocation, modelType , device, n_gpu)
logger.info("Training Model -- DONE")
# logger.info("Testing Model")