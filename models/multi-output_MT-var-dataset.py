#!/usr/bin/env python

from __future__ import print_function

from logging import info

import random
import numpy as np
import datetime
from ltlib.evaluation import conll_summary, per_type_summary, is_iob_tagging

from keras.models import Model
from keras.layers import Input, Embedding, merge, Flatten, Dense
from keras.layers import Reshape, Convolution2D, Dropout

from ltlib import filelog
from ltlib import conlldata
from ltlib import viterbi

from ltlib.features import NormEmbeddingFeature, SennaCapsFeature
from ltlib.features import windowed_inputs
from ltlib.callbacks import token_evaluator, EpochTimer
from ltlib.layers import concat, inputs_and_embeddings
from ltlib.settings import cli_settings, log_settings
from ltlib.optimizers import get_optimizer
from ltlib.util import unique

from config import Defaults

config = cli_settings(['datadir', 'datasets','wordvecs'], Defaults)
assert len(config.filter_nums) == len(config.filter_sizes)

datasets = config.datasets.split(',')

data = []
max_fs = []
max_vfs = []

for ind, dataset in enumerate(datasets):
    data_path = config.datadir + '/' + dataset
    data.append(conlldata.load_dir(data_path, config))
    max_fs.append((0.0,0.0))
    max_vfs.append((0.0,0.0))
    
max_y = 0
for ind, ds in enumerate(data):
    y_shape = ds.train.tokens.targets.shape
    if y_shape[1] > max_y:
        max_y = y_shape[1]

all_vocab = set()
for ind, dataset in enumerate(datasets):
    all_vocab = set().union(*[all_vocab, data[ind].vocabulary])

   
w2v = NormEmbeddingFeature.from_file(config.wordvecs,
                                    max_rank=config.max_vocab_size,
                                    vocabulary=all_vocab,
                                    name='words')
features = [w2v]
if config.word_features:
    features.append(SennaCapsFeature('caps'))

for ind, dataset in enumerate(datasets):
    data[ind].tokens.add_features(features)
    data[ind].tokens.add_inputs(windowed_inputs(config.window_size, features))

# Log word vector feature stat summary
info('{}: {}'.format(config.wordvecs, w2v.summary()))

inputs, embeddings = inputs_and_embeddings(features, config)

# Combine and reshape for convolution
seq = concat(embeddings)
cshape = (config.window_size, sum(f.output_dim for f in features))
seq = Reshape((1,) + cshape)(seq)

# Convolutions
conv_outputs = []
for filter_size, filter_num in zip(config.filter_sizes, config.filter_nums):
    conv = Convolution2D(filter_num, filter_size, cshape[1],activation='relu')(seq)
    cout = Flatten()(conv)
    conv_outputs.append(cout)
seq = concat(conv_outputs)

for size in config.hidden_sizes:
    seq = Dense(size, activation=config.hidden_activation)(seq)
seq = Dropout(config.output_drop_prob)(seq)

#Create private outputs
outs = []
for ind, dataset in enumerate(datasets):
    #outs.append(Dense(data[ind].tokens.target_dim, activation='softmax')(seq))
    outs.append(Dense(max_y, activation='softmax')(seq))

model = Model(input=inputs, output=outs)
optimizer = get_optimizer(config)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

ind_bounds = [(0, len(ds.train.tokens.inputs['words'])) for ds in data]
if config.percent_keep < 1.0:
    percnt_keep = config.percent_keep
    #Only limit first dataset. Other remains full
    for ind, ds in enumerate(data[:1]):
        amt_keep = len(ds.train.tokens.inputs['words']) * percnt_keep
        print("%s: Total: %s. Keeping: %s" % (datasets[ind], len(ds.train.tokens.inputs['words']), amt_keep))
        start = random.randrange(int(len(ds.train.tokens.inputs['words']) - amt_keep + 1))
        end = int(start + amt_keep)
        ind_bounds[ind] = (start,end)
        
x_batch = []
y_batch = []
concatenated = True
for ind, ds in enumerate(data):
    start = ind_bounds[ind][0]
    end = ind_bounds[ind][1]
    x_batch.append(data[ind].train.tokens.inputs['words'][start:end])
    
    out_labels = [np.zeros((dataset.train.tokens.targets[ind_bounds[ind_][0]:ind_bounds[ind_][1]].shape[0], max_y)) for ind_, dataset in enumerate(data)]
    y_ = data[ind].train.tokens.targets[start:end]
    if y_.shape[1] < max_y:
        y_ = np.concatenate([y_, np.zeros((y_.shape[0],max_y - y_.shape[1]))], axis=1)
    
    out_labels[ind] = y_ #data[ind].train.tokens.targets[start:end]
    try:
        y_batch.append(np.concatenate(out_labels, axis=0))
    except:
        print("Cannot concatenate Datasets. Expect slower run time.")
        concatenated = False
        break
    
if concatenated:
    x_batch = np.concatenate(x_batch, axis=0)

"""
for ind, ds in enumerate(data):
    x_batch.append(ds.train.tokens.inputs['words'])
    
    #out_labels = [np.zeros(data[ind_].train.tokens.targets.shape) for ind_, dataset in enumerate(datasets)]
    #out_labels[ind] = data[ind].train.tokens.targets
    
    out_labels = [np.zeros((data[ind_].train.tokens.targets.shape[0], max_y)) for ind_, dataset in enumerate(datasets)]
    y_ = data[ind].train.tokens.targets
    if y_.shape[1] < max_y:
        y_ = np.concatenate([y_, np.zeros((y_.shape[0],max_y - y_.shape[1]))], axis=1)
    out_labels[ind] = y_
    
    try:
        y_batch.append(np.concatenate(out_labels, axis=0))
    except:
        print("Cannot concatenate Datasets. Expect slower run time.")
        concatenated = False
        break
    
if concatenated:
    x_batch = np.concatenate(x_batch, axis=0)
"""

time_str = datetime.datetime.now().isoformat()
print("Started training at: %s" % time_str)
for step in range(1, config.train_steps + 1):
    
    if concatenated:
        start = random.randrange(len(x_batch))
        end = start + config.batch_size
        x = x_batch[start:end]
        y = [y_[start:end] for y_ in y_batch]
    else:
        #Untested in var dataset context
        y = []
        data_ind = random.randrange(len(data))
        #Start create batch
        start = random.randrange(len(data[data_ind].train.tokens.inputs['words']))
        end = start + config.batch_size
        x = data[data_ind].train.tokens.inputs['words'][start:end]
        #End create batch
        #Keras requires labels for all outputs. Create dummy outputs for ones not being trained.
        out_labels = []
        out_labels = [np.zeros(data[ind].train.tokens.targets[0:len(x)].shape) for ind, dataset in enumerate(datasets)]
        out_labels[data_ind] = data[data_ind].train.tokens.targets[start:end]
        for ind, ol in enumerate(out_labels):
            if len(y) == len(data):
                y[ind] = np.concatenate([y[ind], ol], axis=0)
            else:
                y.append(ol)
    
    model.train_on_batch(x, y)
    
    if step % config.evaluate_every == 0 and step >= config.evaluate_min:
        time_str = datetime.datetime.now().isoformat()
        info("\nEvaluating. Time: {}. Step: {}".format(time_str, step))
        for eval_ind, dataset in enumerate(datasets[:1]):
            info("Dataset: {}".format(datasets[eval_ind]))
            predictions = model.predict(data[eval_ind].devel.tokens.inputs)
            if type(predictions).__name__ != 'list':
                predictions = [predictions]
            pred = predictions[eval_ind]
            
            if is_iob_tagging(unique(data[eval_ind].tokens.target_strs)):
                data[eval_ind].devel.tokens.set_predictions(pred)
                summary = conll_summary(data[eval_ind].devel.sentences)
                #Track Maxes
                f_score = summary.split(':')[2].split('%')[0].strip()
                try:
                    f_score = float(f_score)
                except:
                    print("%s is not floatable!" % f_score)
                if f_score > max_fs[eval_ind][0]:
                    max_fs[eval_ind] = (f_score, max_fs[eval_ind][0])
                elif f_score > max_fs[eval_ind][1]:
                    max_fs[eval_ind] = (max_fs[eval_ind][0], f_score)
                #End Track Maxes    
                info("{}".format(summary))
                info("Max Fs: {}".format(str(max_fs[eval_ind])))
                if config.viterbi:
                    vmapper = viterbi.TokenPredictionMapper(data[eval_ind].train.sentences)
                    data[eval_ind].devel.sentences.map_predictions(vmapper)
                    info(vmapper.summary())
                    vsummary = conll_summary(data[eval_ind].devel.sentences)
                    #Track Maxes
                    vf_score = vsummary.split(':')[2].split('%')[0].strip()
                    try:
                        vf_score = float(vf_score)
                    except:
                        print("Viterbi %s is not floatable!" % vf_score)
                    if vf_score > max_vfs[eval_ind][0]:
                        max_vfs[eval_ind] = (vf_score, max_vfs[eval_ind][0])
                    elif vf_score > max_vfs[eval_ind][1]:
                        max_vfs[eval_ind] = (max_vfs[eval_ind][0], vf_score)
                    #End Track Maxes    
                    info("{}".format(vsummary))
                    info("Max Viterbi Fs: {}".format(str(max_vfs[eval_ind])))
            else:
                info("{}".format(per_type_summary(data[eval_ind].tokens)))
