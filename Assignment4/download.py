from __future__ import print_function
import os
import sys
import json
import zipfile
import subprocess
from six.moves import urllib

def download_mnist(dirpath):
    data_dir = os.path.join(dirpath, 'mnist')
    if os.path.exists(data_dir):
        print('Found MNIST - skip')
        return
    else:
        os.mkdir(data_dir)
    url_base = 'http://yann.lecun.com/exdb/mnist/'
    file_names = ['train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz','t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz']
    for file_name in file_names:
        url = (url_base+file_name).format(**locals())
        print(url)
        out_path = os.path.join(data_dir,file_name)
        cmd = ['curl', url, '-o', out_path]
        print('Downloading ', file_name)
        subprocess.call(cmd)
        cmd = ['gzip', '-d', out_path]
        print('Decompressing ', file_name)
        subprocess.call(cmd)

def prepare_data_dir(path = './data'):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    prepare_data_dir()
    download_mnist('./data')
