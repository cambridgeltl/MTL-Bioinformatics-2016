#!/usr/bin/env python

from __future__ import print_function

from logging import info

from keras.models import Model
from keras.layers import Input, Embedding, merge, Flatten, Dense
from keras.layers import Dropout

from ltlib import filelog
from ltlib import conlldata
from ltlib import viterbi

from ltlib.features import NormEmbeddingFeature, SennaCapsFeature
from ltlib.features import windowed_inputs
from ltlib.callbacks import token_evaluator, EpochTimer
from ltlib.layers import concat, inputs_and_embeddings
from ltlib.settings import cli_settings, log_settings
from ltlib.optimizers import get_optimizer
from ltlib.output import save_token_predictions

from baseline_config import Defaults

config = cli_settings(['datadir', 'wordvecs'], Defaults)

data = conlldata.load_dir(config.datadir, config)

vmapper = viterbi.get_prediction_mapper(data.train.sentences, config)

w2v = NormEmbeddingFeature.from_file(config.wordvecs,
                                     max_rank=config.max_vocab_size,
                                     vocabulary=data.vocabulary,
                                     name='words')
features = [w2v]
if config.word_features:
    features.append(SennaCapsFeature(name='caps'))

data.tokens.add_features(features)
data.tokens.add_inputs(windowed_inputs(config.window_size, features))

# Log word vector feature stat summary
info('{}: {}'.format(config.wordvecs, w2v.summary()))

inputs, embeddings = inputs_and_embeddings(features, config)

seq = concat(embeddings)
seq = Flatten()(seq)
for size in config.hidden_sizes:
    seq = Dense(size, activation=config.hidden_activation)(seq)
seq = Dropout(config.output_drop_prob)(seq)
out = Dense(data.tokens.target_dim, activation='softmax')(seq)
model = Model(input=inputs, output=out)

optimizer = get_optimizer(config)
model.compile(loss='categorical_crossentropy', optimizer=optimizer,
              metrics=['accuracy'])

callbacks = [
    EpochTimer(),
    token_evaluator(data.train, config=config),
    token_evaluator(data.test, mapper=vmapper, config=config),
]

model.fit(
    data.train.tokens.inputs,
    data.train.tokens.targets,
    callbacks=callbacks,
    batch_size=config.batch_size,
    nb_epoch=config.epochs,
    verbose=config.verbosity
)

save_token_predictions(data.devel, model, conlldata.write)
save_token_predictions(data.test, model, conlldata.write)
