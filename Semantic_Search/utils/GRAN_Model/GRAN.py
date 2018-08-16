import theano
import numpy as np
import lasagne
import pickle
import sys
from theano import tensor as T
from theano import config

from Semantic_Search.utils.GRAN_Model.lasagne_gran_layer import lasagne_gran_layer
from Semantic_Search.utils.GRAN_Model.lasagne_sum_layer import lasagne_sum_layer
from Semantic_Search.utils.GRAN_Model.lasagne_average_layer import lasagne_average_layer
from Semantic_Search.utils.GRAN_Model.lasagne_gran_layer_nooutput_layer import lasagne_gran_layer_nooutput_layer


def check_quarter(idx, n):
    if idx == round(n / 4.) or idx == round(n / 2.) or idx == round(3 * n / 4.):
        return True
    return False


class models(object):

    def prepare_data(self, list_of_seqs):
        lengths = [len(s) for s in list_of_seqs]
        n_samples = len(list_of_seqs)
        maxlen = np.max(lengths)
        x = np.zeros((n_samples, maxlen)).astype('int32')
        x_mask = np.zeros((n_samples, maxlen)).astype(theano.config.floatX)
        for idx, s in enumerate(list_of_seqs):
            x[idx, :lengths[idx]] = s
            x_mask[idx, :lengths[idx]] = 1.
        x_mask = np.asarray(x_mask, dtype=config.floatX)
        return x, x_mask

    def save_params(self, fname):
        f = open(fname, 'wb')
        values = lasagne.layers.get_all_param_values(self.layer)
        pickle.dump(values, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def get_minibatches_idx(self, n, minibatch_size, shuffle=False):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)


    def __init__(self, We_initial, params):

        initial_We = theano.shared(np.asarray(We_initial, dtype=config.floatX))
        We = theano.shared(np.asarray(We_initial, dtype=config.floatX))

        self.dropout = params['dropout']
        self.word_dropout = params['word_dropout']

        g1batchindices = T.imatrix()

        g1mask = T.matrix()

        l_in = lasagne.layers.InputLayer((None, None))
        l_mask = lasagne.layers.InputLayer(shape=(None, None))
        l_emb = lasagne.layers.EmbeddingLayer(l_in, input_size=We.get_value().shape[0],
                                              output_size=We.get_value().shape[1], W=We)

        if params['dropout'] > 0:
            l_emb = lasagne.layers.DropoutLayer(l_emb, params['dropout'])
        elif params['word_dropout'] > 0:
            l_emb = lasagne.layers.DropoutLayer(l_emb, ['word_dropout'], shared_axes=(2,))



        if params['model'] == "gran":
            if params['outgate']:
                l_lstm = lasagne_gran_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                            mask_input=l_mask, gran_type=params['gran_type'])
            else:
                l_lstm = lasagne_gran_layer_nooutput_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                                           mask_input=l_mask, gran_type=params['gran_type'])

            if params['gran_type'] == 1 or params['gran_type'] == 2:
                l_out = lasagne_average_layer([l_lstm, l_mask], tosum=False)
            else:
                l_out = lasagne.layers.SliceLayer(l_lstm, -1, 1)

        elif params['model'] == "bigran":
            if params['outgate']:
                l_lstm = lasagne_gran_layer_nooutput_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                            mask_input=l_mask, gran_type=params['gran_type'])
                l_lstmb = lasagne_gran_layer_nooutput_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                             mask_input=l_mask, backwards=True)
            else:
                l_lstm = lasagne_gran_layer_nooutput_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                                           mask_input=l_mask, gran_type=params['gran_type'])
                l_lstmb = lasagne_gran_layer_nooutput_layer(l_emb, params['dim'], peepholes=True, learn_init=False,
                                                            mask_input=l_mask, backwards=True)

            if not params['sumlayer']:
                l_concat = lasagne.layers.ConcatLayer([l_lstm, l_lstmb], axis=2)
                l_out = lasagne.layers.DenseLayer(l_concat, params['dim'], num_leading_axes=-1,
                                                  nonlinearity=lasagne.nonlinearities.tanh)
                l_out = lasagne_average_layer([l_out, l_mask], tosum=False)
            else:
                l_out = lasagne_sum_layer([l_lstm, l_lstmb])
                l_out = lasagne_average_layer([l_out, l_mask], tosum=False)
        else:
            print("Invalid model specified. Exiting.")
            sys.exit(0)

        self.final_layer = l_out

        embg1 = lasagne.layers.get_output(l_out, {l_in: g1batchindices, l_mask: g1mask}, deterministic=False)

        self.encoding_function = theano.function([g1batchindices, g1mask], embg1)


    def scramble(self, t, words):
        t.populate_embeddings_scramble(words)
