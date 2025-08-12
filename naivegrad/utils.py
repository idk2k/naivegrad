import os
import numpy as np

def mask_like(like, mask_inx, mask_value = 1.0):
    mask = np.zeros_like(like).reshape(-1)
    mask[mask_inx] = mask_value
    return mask.reshape(like.shape)

def initialize_layer_uniform(*x):
    ret = np.random.uniform(-1., 1., size=x) / np.sqrt(np.prod(x))
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

# matlib functions to speedup convs

def im2col(x, H, W):
    bs,cin,oy,ox = x.shape[0], x.shape[1], x.shape[2]-(H-1), x.shape[3]-(W-1)

    tx = np.empty((oy, ox, bs, cin*W*H), dtype=x.dtype)
    for Y in range(oy):
        for X in range(ox):
            tx[Y, X] = x[:, :, Y:Y+H, X:X+W].reshape(bs, -1)
    return tx.reshape(-1, cin*W*H)

def col2im(tx, H, W, OY, OX):
    oy, ox = OY-(H-1), OX-(W-1)
    bs = tx.shape[0] // (oy * ox)
    cin = tx.shape[1] // (H * W)
    tx = tx.reshape(oy, ox, bs, cin, H, W)

    x = np.zeros((bs, cin, OY, OX), dtype=tx.dtype)
    for Y in range(oy):
        for X in range(ox):
            x[:, :, Y:Y+H, X:X+W] += tx[Y, X]
    return x