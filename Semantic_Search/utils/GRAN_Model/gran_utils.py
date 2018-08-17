import numpy as np
from random import randint
from random import choice
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from config import DEBUG_ALGORITHM_TESTING


def lookupIDX(words,w):
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#","")
    if w in words:
        return words[w]
    else:
        return None

def get_wordmap(textfile):
    words={}
    We = []
    f = open(textfile,'r')
    lines = f.readlines()
    for (n,i) in enumerate(lines):
        i=i.split()
        j = 1
        v = []
        while j < len(i):
            v.append(float(i[j]))
            j += 1
        words[i[0]]=n
        We.append(v)
    return (words, np.array(We))

def get_pairs_rand(d, idx):
    wpick = None
    ww = None
    while(wpick == None or (idx == ww)):
        ww = choice(d)
        ridx = randint(0,1)
        wpick = ww[ridx]
    return wpick

def get_pairs_mix(d, idx, maxpair):
    r1 = randint(0,1)
    if r1 == 1:
        return maxpair
    else:
        return get_pairs_rand(d, idx)

def get_pairs_fast(d, type):
    X = []
    T = []
    pairs = []
    for i in range(len(d)):
        (p1,p2) = d[i]
        X.append(p1.representation)
        X.append(p2.representation)
        T.append(p1)
        T.append(p2)

    arr = pdist(X,'cosine')
    arr = squareform(arr)

    for i in range(len(arr)):
        arr[i,i]=1
        if i % 2 == 0:
            arr[i,i+1] = 1
        else:
            arr[i,i-1] = 1

    arr = np.argmin(arr,axis=1)
    for i in range(len(d)):
        (t1,t2) = d[i]
        p1 = None
        p2 = None
        if type == "MAX":
            p1 = T[arr[2*i]]
            p2 = T[arr[2*i+1]]
        if type == "RAND":
            p1 = get_pairs_rand(d, i)
            p2 = get_pairs_rand(d, i)
        if type == "MIX":
            p1 = get_pairs_mix(d, i, T[arr[2 * i]])
            p2 = get_pairs_mix(d, i, T[arr[2 * i + 1]])
        pairs.append((p1,p2))
    return pairs


def get_seq(p1, words):
    p1 = p1.split()
    X1 = []
    for i in p1:
        index = lookupIDX(words,i)
        if index is None:
            if DEBUG_ALGORITHM_TESTING:
                continue
            else:
                return None
        X1.append(index)
    return X1
