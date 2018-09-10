import argparse

import lasagne
import numpy as np
import _pickle as cPickle

from utils.GRAN_Model.GRAN import models
from utils.GRAN_Model.gran_utils import get_wordmap, lookupIDX

parser = argparse.ArgumentParser()
parser.add_argument("-LC", help="Regularization on composition parameters", type=float, default=0.)
parser.add_argument("-LW", help="Regularization on embedding parameters", type=float, default=0.)
parser.add_argument("-outfile", help="Name of output file")
parser.add_argument("-batchsize", help="Size of batch", type=int, default=100)
parser.add_argument("-dim", help="Dimension of model", type=int, default=300)
parser.add_argument("-wordfile", help="Word embedding file")
parser.add_argument("-save", help="Whether to pickle model", default="False")
parser.add_argument("-margin", help="Margin in objective function", type=float, default=0.4)
parser.add_argument("-samplingtype", help="Type of Sampling used: MAX, MIX, or RAND", default="MAX")
parser.add_argument("-evaluate", help="Whether to evaluate the model during training", default="False")
parser.add_argument("-epochs", help="Number of epochs in training", type=int, default=10)
parser.add_argument("-eta", help="Learning rate", type=float, default=0.001)
parser.add_argument("-learner", help="Either AdaGrad or Adam", default="adam")
parser.add_argument("-outgate", help="Whether to have an outgate for models using LSTM",default="True")
parser.add_argument("-model", help="Which model to use between (bi)lstm, (bi)lstmavg, (bi)gran, or wordaverage")
parser.add_argument("-mode", help="Train on SimpWiki (default) or equivalent amount of tokens from PPDB (set to ppdb)", default="simpwiki")
parser.add_argument("-scramble", type=float, help="Rate of scrambling", default=0.5)
parser.add_argument("-dropout", type=float, help="Dropout rate", default=0.)
parser.add_argument("-word_dropout", type=float, help="Word dropout rate", default=0.)
parser.add_argument("-gran_type", type=int,  help="Type of GRAN model", default=1)
parser.add_argument("-sumlayer", help="Whether to use sum layer for bi-directional recurrent networks", default="False")
parser.add_argument("-max", type=int, help="Maximum number of examples to use (<= 0 means use all data)", default=0)
parser.add_argument("-loadmodel", help="Name of pickle file containing model", default=None)

params = parser.parse_args()
params.save = False
params.evaluate = False
params.outgate = True
params.learner = lasagne.updates.adam
params.model = 'gran'



def get_seq(p1, words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words,i))
    return X1


def encode_sentence(sen, model, words):
    seq = []
    X1 = get_seq(sen, words)
    seq.append(X1)
    x1, m1 = model.prepare_data(seq)
    embedding = model.encoding_function(x1, m1)

    return embedding

wordfile = '/home/sw/NLP/models/gran_model/paragram_sl999_small.txt'
modelfile = '/home/sw/NLP/models/gran_model/gran.pickle'

params = {'dropout': 0.0, 'word_dropout': 0.0, 'model': 'gran', 'outgate': True, 'gran_type': 1,
             'dim': 300, 'sumlayer': False}

(words, We) = get_wordmap(wordfile)
model = models(We, params)
base_params = cPickle.load(open(modelfile, 'rb'), encoding='iso-8859-1')
lasagne.layers.set_all_param_values(model.final_layer, base_params)

vec = encode_sentence('hello, the world', model, words)

print(vec)

