import time
import re

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

import logging


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        # print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        logging.info('%r %2.2f sec' % (method.__name__, te - ts))
        return result

    return timed


def des_sort(scores):
    des_score = sorted(enumerate(scores), key=lambda item: -item[1])
    # logging.info(des_score)
    return des_score


def split_by_capital(phase):
    splits = []
    if phase.isupper():
        splits = [phase]
    else:
        # split by capital letter
        splits = re.findall('[A-Z][^A-Z]*', phase)
        splits = [phase[0:phase.find(splits[0])]] + splits if len(splits) > 0 else [phase]
    return splits


def split_phase(phase):
    splits = []

    if '-' in phase:
        splits = phase.split('-')
    else:
        splits = [phase]

    pre_splits = []
    for a_split in splits:
        if '_' in a_split:
            pre_splits = pre_splits + a_split.split('_')
        else:
            pre_splits.append(a_split)
    splits = pre_splits

    pre_splits = []
    for a_split in splits:
        if ' ' in a_split:
            pre_splits = pre_splits + a_split.split(' ')
        else:
            pre_splits.append(a_split)
    splits = pre_splits

    pre_splits = []
    for a_split in splits:
        pre_splits = pre_splits + split_by_capital(a_split)

    '''
    #gensim.utils.simple_preprocess will do the split by " ", "-", remove sysmbols and a little stopwords
    #**but not "_", the above codes should be added if need to split by "_"
    '''
    splits.clear()
    for one_unit in pre_splits:
        splits = splits + simple_preprocess(one_unit)

    # and remove '_|-| '
    splits = [re.sub('_|-| |\(|\)', '', word) for word in splits]

    splits = [word for word in splits if word not in STOPWORDS]

    return splits


if __name__ == '__main__':
    str = "it is time for WorldCup! Isn't it great? I want a-cup-of-coffee, do you want a hello_Kitty!"
    # str = split_doc(str)
    str = split_phase(str)
    print(str)
