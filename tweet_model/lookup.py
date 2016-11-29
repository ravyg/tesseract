#%matplotlib inline
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE

class lookup(object):

  def __init__(self):
    """Constructor"""
    # self.generator = Generator()
    # self.decoder = Decoder()
    # print "READY"

  def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
      filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
      print('Found and verified %s' % filename)
    else:
      print(statinfo.st_size)
      raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

  def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
      data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
    
  words = read_data(filename)
  print('Data size %d' % len(words))