'''gensim Word2Vec'''
w2v_model_path = '/../models/w2v_model/'
w2v_model_files = ['GoogleNews-vectors-negative300.bin', 'glove.twitter.27B.200d.w2v.txt']
w2v_model_choice = 'google' #or 'glove'


'''fastText'''
fasttext_model_filepath = '/../models/fasttext/wiki.en.bin'

'''infersent'''
infersent_model_filepath = '/../models/infersent_model/infersent{}.pkl'
infersent_version = 2
infersent_wordvec_filepath = ['/../models/infersent_model/glove.840B.300d.txt',
                              '/../models/infersent_model/crawl-300d-2M.vec']

'''gran'''
gran_model_filepath = '/../models/gran_model/gran.pickle'
gran_wordvec_filepath = '/../models/gran_model/paragram_sl999_small.txt'




'''
#
# parameters for weighted average vector algorithm
#
'''
vector_u_cache_file = 'vector_u_for_{}.pkl'
coefficient_a = 1e-3



import logging
import sys

# logger = logging.getLogger('')
# logger.setLevel(logging.INFO)
# fh = logging.FileHandler('/home/sw/NLP/process.log')
# sh = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
# fh.setFormatter(formatter)
# sh.setFormatter(formatter)
# # logger.addHandler(fh)
# logger.addHandler(sh)
