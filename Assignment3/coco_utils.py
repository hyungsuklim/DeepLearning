# This library is used for Assignment3_Part2_ImageCaptioning

import os, json
import numpy as np
import h5py
import urllib, os, tempfile
from scipy.misc import imread
import matplotlib.pyplot as plt
import nltk

NUMVAL = 10000

def load_coco_data(base_dir='coco/coco_captioning',
									 max_train=None,
									 pca_features=True):
	data = {}
	caption_file = os.path.join(base_dir, 'train2014_captions.h5')
	with h5py.File(caption_file, 'r') as f:
		for k, v in f.items():
			data[k] = np.asarray(v)

	if pca_features:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
	else:
		train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
	with h5py.File(train_feat_file, 'r') as f:
		data['features'] = np.asarray(f['features'])

	dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
	with open(dict_file, 'r') as f:
		dict_data = json.load(f)
		for k, v in dict_data.items():
			data[k] = v

	train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
	with open(train_url_file, 'r') as f:
		train_urls = np.asarray([line.strip() for line in f])
	data['urls'] = train_urls

	num_train = data['train_captions'].shape[0]
	mask = np.arange(num_train-NUMVAL,num_train) 
	data['val_captions'] = data['train_captions'][mask]
	data['val_image_idxs'] = data['train_image_idxs'][mask]
#	data['val_features'] = data['train_features']
#	data['val_urls'] = data['train_urls']

# Maybe subsample the training data
	if max_train is not None:
		num_train = data['train_captions'].shape[0]
		mask = np.random.randint(num_train-NUMVAL, size=max_train)
		data['train_captions'] = data['train_captions'][mask]
		data['train_image_idxs'] = data['train_image_idxs'][mask]

	return data



def decode_captions(captions, idx_to_word):
	singleton = False
	if captions.ndim == 1:
		singleton = True
		captions = captions[None]
	decoded = []
	N, T = captions.shape
	for i in range(N):
		words = []
		for t in range(T):
			word = idx_to_word[captions[i, t]]
			if word != '<NULL>':
				words.append(word)
			if word == '<END>':
				break
		decoded.append(' '.join(words))
	if singleton:
		decoded = decoded[0]
	return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
	split_size = data['%s_captions' % split].shape[0]
	if batch_size=="All":
		mask = np.arange(split_size)
		print("Total captions:", mask.shape[0])
	elif batch_size > split_size :
		mask = np.random.choice(split_size, split_size, replace=False)
	else : mask = np.random.choice(split_size, batch_size)
	captions = data['%s_captions' % split][mask]
	image_idxs = data['%s_image_idxs' % split][mask]
	image_features = data['features'][image_idxs]
	urls = data['urls'][image_idxs]
	return captions, image_features, urls


def image_from_url(url):
  """
  Read an image from a URL. Returns a numpy array with the pixel data.
  We write the image to a temporary file then read it back. Kinda gross.
  """
  try:
    f = urllib.request.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = imread(fname)
    os.remove(fname)
    return img
  except urllib2.URLError as e:
    print('URL Error: ', e.reason, url)
  except urllib2.HTTPError as e:
    print('HTTP Error: ', e.code, url)


def show_samples(captions, urls, data):
  for i, (caption, url) in enumerate(zip(captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    caption_str = decode_captions(caption, data['idx_to_word'])
    plt.title(caption_str, loc='left')
    plt.show()

def show_predict_samples(gt_captions, pr_captions, urls, data):
  for i, (gt_caption, pr_caption, url) in enumerate(zip(gt_captions, pr_captions, urls)):
    plt.imshow(image_from_url(url))
    plt.axis('off')
    gt_caption_str = decode_captions(gt_caption, data['idx_to_word'])
    pr_caption_str = decode_captions(pr_caption, data['idx_to_word'])
    plt.title('G-Truth:%s\nPredict :%s'% (gt_caption_str,pr_caption_str), loc='left')
    plt.show()


def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ')
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ')
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

