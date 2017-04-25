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
from ltlib.output import save_token_predictions

from config import Defaults

config = cli_settings(['datadir', 'datasets','wordvecs'], Defaults)
assert len(config.filter_nums) == len(config.filter_sizes)

datasets = config.datasets.split(',')

data = []
max_fs = []
max_vfs = []
vmappers = []
for ind, dataset in enumerate(datasets):
    data_path = config.datadir + '/' + dataset
    if ind != 0:
        config.iobes = True
    data.append(conlldata.load_dir(data_path, config))
    max_fs.append((0.0,0.0))
    max_vfs.append((0.0,0.0))
    
all_vocab = set()
for ind, dataset in enumerate(datasets):
    all_vocab = set().union(*[all_vocab, data[ind].vocabulary])

   
for ind, dataset in enumerate(datasets):
    w2v = NormEmbeddingFeature.from_file(config.wordvecs,
                                        max_rank=config.max_vocab_size,
                                        vocabulary=all_vocab,
                                        name='words-%s' % ind)
    features = [w2v]
    if config.word_features:
        features.append(SennaCapsFeature('caps'))


    data[ind].tokens.add_features(features)
    data[ind].tokens.add_inputs(windowed_inputs(config.window_size, features))

    # Log word vector feature stat summary
    info('{}: {}'.format(config.wordvecs, w2v.summary()))

    if ind == 0:
        pos_inputs, pos_embeddings = inputs_and_embeddings(features, config)
        pos_x = concat(pos_embeddings)
    if ind == 1:
        ner_inputs, ner_embeddings = inputs_and_embeddings(features, config)
        ner_x = concat(ner_embeddings)
           
cshapes = []
reshapes = []                              
# Combine and reshape for convolution
pos_cshape = (config.window_size, sum(f.output_dim for f in features))
ner_cshape = (config.window_size, sum(f.output_dim for f in features))
cshapes.append(pos_cshape)
cshapes.append(ner_cshape)

pos_reshape = Reshape((1,) + (pos_cshape), name='pos-reshape')(pos_x)
ner_reshape = Reshape((1,) + (ner_cshape), name='ner-reshape')(ner_x)
reshapes.append(pos_reshape)
reshapes.append(ner_reshape)

# Convolutions
conv_outputs = []
fully_connected = []
dropout = []
for ind, dataset in enumerate(datasets):
    conv_outputs.append([])
    for filter_size, filter_num in zip(config.filter_sizes, config.filter_nums):
        conv = Convolution2D(filter_num, filter_size, cshapes[ind][1], activation='relu', name='convolution-%d-%d' % (ind,filter_size))(reshapes[ind])
        flatten = Flatten(name='flatten-%d-%d' % (ind,filter_size))(conv)
        conv_outputs[ind].append(flatten)
        
    seq = concat(conv_outputs[ind])
      
    for size in config.hidden_sizes:
        fully_connected.append(Dense(size, activation=config.hidden_activation, name='dense-1-%d' % ind)(seq))
    dropout.append(Dropout(config.output_drop_prob, name='dropout-%d' % ind)(fully_connected[ind]))

pos_dense_out = Dense(data[0].tokens.target_dim, activation='softmax', name='pos-dense-out')(dropout[0])

ner_merged = merge([dropout[0], dropout[1]], mode='concat')
ner_dense_out = Dense(data[1].tokens.target_dim, activation='softmax', name='ner-dense-out')(ner_merged)

pos_model = Model(input=pos_inputs, output=pos_dense_out)
ner_model = Model(input=pos_inputs + ner_inputs, output=ner_dense_out)

pos_model.compile(optimizer=get_optimizer(config), loss='categorical_crossentropy', metrics=['accuracy'])
ner_model.compile(optimizer=get_optimizer(config), loss='categorical_crossentropy', metrics=['accuracy'])

models = [pos_model, ner_model]

time_str = datetime.datetime.now().isoformat()
print("Started training at: %s" % time_str)

for ind, ds in enumerate(data):
    for ep in range(1, config.epochs + 1):
        percnt_keep = config.percent_keep
        amt_keep = len(ds.train.tokens.inputs['words-%s' % ind]) * percnt_keep
        print("Total: %s. Keeping: %s" % (len(ds.train.tokens.inputs['words-%s' % ind]), amt_keep))
        start = random.randrange(int(len(ds.train.tokens.inputs['words-%s' % ind]) - amt_keep) + 1)
        end = int(start + amt_keep)
        x = ds.train.tokens.inputs['words-%s' % ind][start:end]
        if ind > 0:
            x = [x, x]
        models[ind].fit(
            x,
            ds.train.tokens.targets[start:end],
            batch_size=config.batch_size,
            nb_epoch=1,
            verbose=config.verbosity
        )
        
        time_str = datetime.datetime.now().isoformat()
        info("\nEvaluating. Time: {}. Epoch: {}".format(time_str, ep))
        
        info("Dataset: {}".format(datasets[ind]))
        if ind > 0:
            predictions = models[ind].predict([ds.test.tokens.inputs['words-%s' % ind], ds.test.tokens.inputs['words-%s' % ind]])
        else:
            predictions = models[ind].predict(ds.test.tokens.inputs['words-%s' % ind])
        
        if is_iob_tagging(unique(ds.tokens.target_strs)):
            ds.test.tokens.set_predictions(predictions)
            summary = conll_summary(ds.test.sentences)
            #Track Maxes
            f_score = summary.split(':')[2].split('%')[0].strip()
            try:
                f_score = float(f_score)
            except:
                print("%s is not floatable!" % f_score)
            if f_score > max_fs[ind][0]:
                max_fs[ind] = (f_score, max_fs[ind][0])
                save_token_predictions(data[eval_ind].test, model, conlldata.write)
            elif f_score > max_fs[ind][1]:
                max_fs[ind] = (max_fs[ind][0], f_score)
                save_token_predictions(data[eval_ind].test, model, conlldata.write)
            #End Track Maxes    
            info("{}".format(summary))
            info("Max Fs: {}".format(str(max_fs[ind])))
            if config.viterbi:
                vmapper = viterbi.TokenPredictionMapper(ds.train.sentences)
                ds.test.sentences.map_predictions(vmapper)
                info(vmapper.summary())
                vsummary = conll_summary(ds.test.sentences)
                #Track Maxes
                vf_score = vsummary.split(':')[2].split('%')[0].strip()
                try:
                    vf_score = float(vf_score)
                except:
                    print("Viterbi %s is not floatable!" % vf_score)
                if vf_score > max_vfs[ind][0]:
                    max_vfs[ind] = (vf_score, max_vfs[ind][0])
                    save_token_predictions(data[eval_ind].test, model, conlldata.write, vmapper)
                elif vf_score > max_vfs[ind][1]:
                    max_vfs[ind] = (max_vfs[ind][0], vf_score)
                    save_token_predictions(data[eval_ind].test, model, conlldata.write, vmapper)
                #End Track Maxes    
                info("{}".format(vsummary))
                info("Max Viterbi Fs: {}".format(str(max_vfs[ind])))
        else:
            ds.test.tokens.set_predictions(predictions)
            info("{}".format(per_type_summary(ds.test.tokens)))
