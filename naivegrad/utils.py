import os
import numpy as np

def initialize_layer_uniform(m, h):
    ret = np.random.uniform(-1., 1., size=(m, h)) / np.sqrt(m * h)
    return ret.astype(np.float32)

def fetch_mnist():
    def fetch(url):
        import requests, gzip, os, hashlib, numpy
        if os.name == 'nt':
            #print(f"path: {os.getcwd()}\\tmp")
            fp = os.path.join(os.getcwd() + "\\tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        else:
            fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                dat = f.read()
        else:
            with open(fp, "wb") as f:
                dat = requests.get(url).content
                f.write(dat)
        return numpy.frombuffer(gzip.decompress(dat), dtype=numpy.uint8).copy()
    
    # urls are fixed, cause LeCunn dont maintain his ds
    # https://github.com/pytorch/vision/issues/3549
    X_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
    Y_train = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28))
    Y_test = fetch("https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test