
'''wiki data'''
wiki_dump_file = '/home/sw/NLP/wiki_data/enwiki-20180601-pages-articles-multistream.xml.bz2'
wiki_corups_file = '/home/sw/NLP/wiki_data/enwiki-20180601.corups'


'''gensim NLP'''
nlp_model_path = '/home/sw/NLP/wiki_data/model'
wordids_file = '_wordids.txt'
corups_file = '_tfidf.mm'
lsi_model_file = 'model.lsi'
lda_model_file = 'model.lda'

'''gensim Word2Vec'''
w2v_model_path = '/home/sw/NLP/models/w2v_model/'
w2v_model_files = ['GoogleNews-vectors-negative300.bin',
               'glove.840B.300d.w2v.txt',
               'glove.twitter.27B.50d.w2v.txt', 'glove.twitter.27B.100d.w2v.txt', 'glove.twitter.27B.200d.w2v.txt']
w2v_model_choice = 1
w2v_stopwords_path = "./data/stopwords_en.txt"

'''gensim Doc2Vec'''
d2v_model_path = '/home/sw/NLP/models/d2v_model/d2v_model_v1'
d2v_model2_path = '/home/sw/NLP/models/d2v_model/d2v_model_v2'
dbow_model_file = 'model.dbow'
dm_model_file = 'model.dm'

'''fastText'''
fasttext_model_filepath = '/Users/kun/Documents/py/thingin_recommender/Semantic_Search/models/fasttext/wiki.en.bin'



TOPICS = 400

#
# import logging, sys
#
# logger = logging.getLogger('')
# logger.setLevel(logging.INFO)
# fh = logging.FileHandler('/home/sw/NLP/process.log')
# sh = logging.StreamHandler(sys.stdout)
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
# fh.setFormatter(formatter)
# sh.setFormatter(formatter)
# logger.addHandler(fh)
# logger.addHandler(sh)