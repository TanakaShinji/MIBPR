import numpy as np
import _pickle as pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from allennlp.modules.elmo import Elmo, batch_to_ids
from tqdm import tqdm
#w2v_fileはhttps://nlp.stanford.edu/projects/glove/からダウンロードする
w2v_file = "maindata_256996/glove.840B.300d.txt"
w2v = None

def vectorize_tagged_items(item_features, item_feature_labels, w2v_pickle=None):
	"""
    Stanfordが事前にトレーニングした埋め込みデータ(w2v_file)を使用して今回レコメンドする各アイテムを
    アイテム名の３００次元, 加重平均したものをベクトルとして返す
    名前のないアイテムは平均０，標準偏差１の正規分布からサンプリングしたランダムなベクトルが与えられる
    このvectorize_documentは（all_titleword_vector256996.npy）データを生成するために使用される
    本プログラムのデータを使用するのであれば実装する必要はない

	パラメータ
	------
	item_textは，行は各アイテム,列はアイテムの単語としたcooまたはcsr型のスパース行列を事前に作成する
	item_text_labelsは文字列の配列，；item_textの列に対応する各アイテム名の文字列ラベル
	"""
	global w2v
	items, vocab = get_item_objects(item_text, item_text_labels)
	if w2v_pickle:
		with open(w2v_pickle, 'rb') as f:
			w2v = pickle.load(f)
	if not w2v:
		w2v = load_bin_vec(w2v_file, vocab)
	W, word_idx_map = get_W(w2v)
	item_embeddings = make_document_embeddings(items, W, word_idx_map)
	return items, item_embeddings

def get_item_objects(item_text, item_text_labels):
	item_text = item_text.tocoo()
	vocab = set(item_text_labels)
	items = [None] * item_text.shape[0]
	prev_item_idx = -1
	i = 0
	while i < len(item_text.row):
		item_idx = item_text.row[i]
		while i < len(item_text.row) and item_idx == prev_item_idx:
			tag = item_text_labels[item_text.col[i]]
			items[item_idx]['tags'].append(tag)
			split_tags = tag.split("-")
			for t in split_tags:
				items[item_idx]['text'].append(t)
				items[item_idx]['weights'].append(1)
			i += 1
			prev_item_idx = item_idx
			if i < len(item_text.row):
				item_idx = item_text.row[i]

		if i < len(item_text.row):
			tag = item_text_labels[item_features.col[i]]
			split_tags = tag.split("-")
			items[item_idx] = {
				'text': split_tags,
				'tags': [tag],
				'weights': [1] * len(split_tags)
				}
			i += 1
			prev_item_idx = item_idx
	return items, vocab

def sparse_vectorize_tagged_items(item_text, item_text_labels):
	items, _ = get_item_objects(item_text, item_text_labels)
	return items, item_text.toarray()

def make_document_embeddings(documents, word_vecs, word_idx_map):
	"""
	アイテムの各単語を加重平均で埋め込んだベクトルに変換する
	"""
	data = []
	total = 0
	for doc in documents:
		if doc:
			weights = doc["weights"]
			words = doc["text"]
			word_embeddings = []
			for word in words:
				total += 1
				if word in word_idx_map:
					idx = word_idx_map[word]
					try:
						word_embeddings.append(word_vecs[idx])
					except:
						print(word)
						print(idx)
						print(len(word_embeddings))
						raise
				else:
					word_embeddings.append(np.random.normal(0, 1, (300,)))
			matrix = np.array(word_embeddings, dtype='float32')
			data.append(np.average(matrix, axis=0, weights=weights))
		else:
			data.append(np.random.normal(0, 1, (300,)))
	print(str(total) + " words embedded.")
	return np.array(data)

def load_bin_vec(fname, vocab):
	"""
    クロールしたGloveデータから300次元の単語ベクトルを読み込む
	"""
	word_vecs = {}
	with open(fname, "r", encoding='latin-1') as f:
		for line in f:
			vec_line = line.rstrip().split(' ')
			word = vec_line[0]
			word_vec = np.array([float(vec_line[i]) for i in range(1, len(vec_line))])
			if word in vocab:
				word_vecs[word] = word_vec
			else:
				continue
			if len(word_vecs) == len(vocab):
				break
	return word_vecs

def get_W(word_vecs, k=300):
	"""
	単語行列を取得するメゾット．W[i]はiでインデックス付けされた単語のベクトル．
	"""
	vocab_size = len(word_vecs)
	word_idx_map = dict()
	W = np.zeros(shape=(vocab_size+1, k))            
	W[0] = np.zeros(k)
	i = 1
	for word in word_vecs:
		W[i] = word_vecs[word]
		word_idx_map[word] = i
		i += 1
	return W, word_idx_map
 
 
"""
#ELMoを利用した文章をベクトル変換したデータを作成する
ELMoを利用する場合，事前に下記の実装をipynbファイルなどで行うこと

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo = Elmo(options_file, weight_file, 2, dropout=0)

パラメータ
------
tradesy_dimは，事前に作成した各アイテムに文章の単語を格納した多次元配列
"""
def get_ELMo(tradesy_dim):
    embedding_1024 = []
    for i in tqdm(tradesy_dim[0:len(tradesy_dim)]):
        sentence = i
        character_ids = batch_to_ids(sentence)
        embeddings = elmo(character_ids)
        vector = embeddings["elmo_representations"[0].detach().numpy()
        for j in range(1, len(vector)):
            vector[0] += vector[j]
        vector_sum1 = vector[0]
        vector_sum1 = vector_sum1 / j
    
        for k in range(1, len(vector_sum1)):
            vector_sum1[0] += vector_sum1[k]
        vector_sum2 = vector_sum1[0]
        vector_sum2 = vector_sum2 / k
        embedding_1024.append(vector_sum2)
    return embedding_1024
