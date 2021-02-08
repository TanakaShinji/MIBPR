import numpy as np
from utils import *
from data import *
from sklearn.metrics import roc_auc_score
import time
import cProfile as profile
from vectorize_documents import *

class MIBPR:
	"""
    機械学習のGoogleなどで利用されるランキング学習のペアワイズ手法に行列分解モデルの予測値を加えて最適化する
    アルゴリズム．このモデルは行列分解に画像処理と自然言語と統計的手法を追加している．
    ハイパーパラメータなどはグリッドサーチで探索すること
	"""
	def __init__(self, latent_dim=2, latent_content_dim=20, latent_money_dim=20, latent_text_dim=20, latent_des_dim = 20, random_seed=None, lambda_b=.01, lambda_neg=.01, lambda_pos=.01, lambda_u=.01, lambda_e=0, lambda_f=0, lambda_g=0, lambda_h=0, lambda_theta=.01):
		self.latent_dim = latent_dim
		self.latent_content_dim = latent_content_dim
		self.latent_money_dim = latent_money_dim
		self.latent_text_dim = latent_text_dim
		self.latent_des_dim = latent_des_dim
		self.lambda_theta = lambda_theta
		self.lambda_h = lambda_h
		self.lambda_g = lambda_g
		self.lambda_f = lambda_f
		self.lambda_e = lambda_e
		self.lambda_u = lambda_u
		self.lambda_neg = lambda_neg
		self.lambda_pos = lambda_pos
		self.lambda_b = lambda_b
		self.random_state = np.random.RandomState(seed=random_seed)

	def init_bpr_params(self, X, item_content_features, item_money_features, item_text_features, item_des_features):
		# 潜在的要因 (gamma_u, gamma_i)
		self.latent_users = self.random_state.uniform(0, 1, (X.shape[0], self.latent_dim))
		self.latent_items = self.random_state.uniform(0, 1, (X.shape[1], self.latent_dim))
		# アイテムのバイアス (beta_i)
		self.items_bias = self.random_state.uniform(0, 1, X.shape[1])
		# 視覚的パラメータ
		self.item_content_features = item_content_features
		if self.item_content_features is not None:
			print("item_content_features provided, using VBPR algorithm")
			# ユーザの視覚的潜在要因 (theta_u)
			self.latent_users_content = self.random_state.uniform(0, 1, (self.latent_users.shape[0], self.latent_content_dim))
			# 線形変換したアイテムの視覚的要因 (E=V_i)
			self.content_embedding_matrix = self.random_state.uniform(0, 1, (self.latent_content_dim, item_content_features.shape[1]))
			# 視覚的要因のバイアス
			self.content_bias = self.random_state.uniform(0, 1, item_content_features.shape[1])

		self.item_money_features = item_money_features
		if self.item_money_features is not None:
			print("item_money_features provided, using MVBPR algorithm")
			# ユーザの金額的要因 (M_u)
			self.latent_users_money = self.random_state.uniform(0, 1, (self.latent_users.shape[0], self.latent_money_dim))
			# 線形変換したアイテムの金額的要因 (F=M_i)
			self.money_embedding_matrix = self.random_state.uniform(0, 1, (self.latent_money_dim, item_money_features.shape[1]))
			# 金額的要因のバイアス
			self.money_bias = self.random_state.uniform(0, 1, item_money_features.shape[1])

		self.item_text_features = item_text_features
		if self.item_text_features is not None:
			print("item_text_features provided, using MTVBPR algorithm")
			# ユーザのアイテム単語的要因 (W_u)
			self.latent_users_text = self.random_state.uniform(0, 1, (self.latent_users.shape[0], self.latent_text_dim))
			# 線形変換したアイテムの単語的要因 (G=W_i)
			self.text_embedding_matrix = self.random_state.uniform(0, 1, (self.latent_text_dim, item_text_features.shape[1]))
			# アイテム単語のバイアス
			self.text_bias = self.random_state.uniform(0, 1, item_text_features.shape[1])

		self.item_des_features = item_des_features
		if self.item_des_features is not None:
			print("item_des_features provided, using MIBPR algorithm")
			# ユーザの文章的要因 (theta_u)
			self.latent_users_des = self.random_state.uniform(0, 1, (self.latent_users.shape[0], self.latent_text_dim))
			# 線形変換したアイテムの文章的要因 (G=theta_i)
			self.des_embedding_matrix = self.random_state.uniform(0, 1, (self.latent_des_dim, item_des_features.shape[1]))
			# アイテム文章のバイアス
			self.des_bias = self.random_state.uniform(0, 1, item_des_features.shape[1])            
	def fit_profile(self, X, y, item_content_features=None, item_money_features=None, item_text_features=None, item_des_features=None, epochs=1, lr=.05):
		profile.runctx('self.fit(X, y, item_content_features, item_money_features, item_text_features, item_des_features, epochs, lr)', globals(), locals())

	def fit(self, X, y=None, item_content_features=None, item_money_features=None, item_text_features=None, item_des_features=None, epochs=1, lr=.05, verbose=0):
		"""
		相互作用行列X（スパース）に対して，MIBPRを実行する関数

		パラメータ
		------
		X: トレーニング用のユーザとアイテムの相互作用を表すcoo型のスパース行列(ユーザ群，アイテム群)
		y: テスト用のユーザとアイテムの相互作用を含むcoo型のスパース行列(ユーザ群，アイテム群)
		epochs: int型の実行回数
		lr: float型の学習率
		"""
		# パラメータの初期化
		self.lr = lr
		self.init_bpr_params(X, item_content_features, item_money_features, item_text_features, item_des_features)
		for epoch in range(epochs):
			print("Epoch {}".format(epoch))
			uids, pids, nids = get_triplets(X)
			for i in range(X.getnnz()):
				pos_item_idx = pids[i]
				user_idx = uids[i]
				neg_item_idx = nids[i]
				pred = self.predict(pos_item_idx, user_idx) - self.predict(neg_item_idx, user_idx)
				d_sigmoid = self.sigmoid(-pred)
				# モデルパラメータの更新
				self.update_params(d_sigmoid, user_idx, pos_item_idx, neg_item_idx)
			if epoch % 10 == 0:
				print('Training AUC: {}'.format(self.auc_score(X)))
				print('Test AUC: {}'.format(self.auc_score(y)))

	def update_params(self, d_sigmoid, user_idx, pos_item_idx, neg_item_idx):
		# 潜在的要因の更新
		pos_latent_item = np.copy(self.latent_items[pos_item_idx])
		neg_latent_item = np.copy(self.latent_items[neg_item_idx])
		latent_user = np.copy(self.latent_users[user_idx])
		self.update_param(self.latent_users[user_idx], d_sigmoid, pos_latent_item - neg_latent_item, self.lambda_u)
		self.update_param(self.latent_items[pos_item_idx], d_sigmoid, latent_user, self.lambda_pos)
		self.update_param(self.latent_items[neg_item_idx], d_sigmoid, -latent_user, self.lambda_neg)

		# 潜在的要因バイアスの更新
		self.update_param(self.items_bias[pos_item_idx], d_sigmoid, 1, self.lambda_b)
		self.update_param(self.items_bias[neg_item_idx], d_sigmoid, -1, self.lambda_b)
		if self.item_content_features is not None:
			# 視覚的要因の更新
			latent_user_content = np.reshape(np.copy(self.latent_users_content[user_idx]), (self.latent_users_content[user_idx].shape[0], 1))
			pos_item_feature = np.reshape(np.copy(self.item_content_features[pos_item_idx]), (self.item_content_features[pos_item_idx].shape[0], 1))
			neg_item_feature = np.reshape(np.copy(self.item_content_features[neg_item_idx]), (self.item_content_features[neg_item_idx].shape[0], 1))
			content_embedding_matrix = np.copy(self.content_embedding_matrix)
			self.update_param(self.latent_users_content[user_idx], d_sigmoid, 
				content_embedding_matrix @ (self.item_content_features[pos_item_idx] - self.item_content_features[neg_item_idx]), self.lambda_theta)
			self.update_param(self.content_embedding_matrix, d_sigmoid,
				latent_user_content @ (pos_item_feature - neg_item_feature).T, self.lambda_e)
			# 視覚的要因バイアスの更新
			self.update_param(self.content_bias, d_sigmoid, self.item_content_features[pos_item_idx] - self.item_content_features[neg_item_idx], self.lambda_b)
            
		if self.item_money_features is not None:
			# 金額的要因の更新
			latent_user_money = np.reshape(np.copy(self.latent_users_money[user_idx]), (self.latent_users_money[user_idx].shape[0], 1))
			pos_item_money = np.reshape(np.copy(self.item_money_features[pos_item_idx]), (self.item_money_features[pos_item_idx].shape[0], 1))
			neg_item_money = np.reshape(np.copy(self.item_money_features[neg_item_idx]), (self.item_money_features[neg_item_idx].shape[0], 1))
			money_embedding_matrix = np.copy(self.money_embedding_matrix)
			self.update_param(self.latent_users_money[user_idx], d_sigmoid, 
				money_embedding_matrix @ (self.item_money_features[pos_item_idx] - self.item_money_features[neg_item_idx]), self.lambda_theta)
			self.update_param(self.money_embedding_matrix, d_sigmoid,
				latent_user_money @ (pos_item_money - neg_item_money).T, self.lambda_f)
			# 金額的要因のバイアスを更新
			self.update_param(self.money_bias, d_sigmoid, self.item_money_features[pos_item_idx] - self.item_money_features[neg_item_idx], self.lambda_b)

		if self.item_text_features is not None:
			# 単語的要因の更新
			latent_user_text = np.reshape(np.copy(self.latent_users_text[user_idx]), (self.latent_users_text[user_idx].shape[0], 1))
			pos_item_text = np.reshape(np.copy(self.item_text_features[pos_item_idx]), (self.item_text_features[pos_item_idx].shape[0], 1))
			neg_item_text = np.reshape(np.copy(self.item_text_features[neg_item_idx]), (self.item_text_features[neg_item_idx].shape[0], 1))
			text_embedding_matrix = np.copy(self.text_embedding_matrix)
			self.update_param(self.latent_users_text[user_idx], d_sigmoid, 
				text_embedding_matrix @ (self.item_text_features[pos_item_idx] - self.item_text_features[neg_item_idx]), self.lambda_theta)
			self.update_param(self.text_embedding_matrix, d_sigmoid,
				latent_user_text @ (pos_item_text - neg_item_text).T, self.lambda_g)
			# 単語的要因のバイアスを更新
			self.update_param(self.text_bias, d_sigmoid, self.item_text_features[pos_item_idx] - self.item_text_features[neg_item_idx], self.lambda_b)            

		if self.item_des_features is not None:
			# 文章的要因の更新
			latent_user_des = np.reshape(np.copy(self.latent_users_des[user_idx]), (self.latent_users_des[user_idx].shape[0], 1))
			pos_item_des = np.reshape(np.copy(self.item_des_features[pos_item_idx]), (self.item_des_features[pos_item_idx].shape[0], 1))
			neg_item_des = np.reshape(np.copy(self.item_des_features[neg_item_idx]), (self.item_des_features[neg_item_idx].shape[0], 1))
			des_embedding_matrix = np.copy(self.des_embedding_matrix)
			self.update_param(self.latent_users_des[user_idx], d_sigmoid, 
				des_embedding_matrix @ (self.item_des_features[pos_item_idx] - self.item_des_features[neg_item_idx]), self.lambda_theta)
			self.update_param(self.des_embedding_matrix, d_sigmoid,
				latent_user_des @ (pos_item_des - neg_item_des).T, self.lambda_h)
			# 文章的要因のバイアスを更新
			self.update_param(self.des_bias, d_sigmoid, self.item_des_features[pos_item_idx] - self.item_des_features[neg_item_idx], self.lambda_b)            

	def update_param(self, theta, d_sigmoid, dx_dtheta, reg_coef):
		theta += self.lr * (d_sigmoid * dx_dtheta - reg_coef * theta)

	def predict(self, item_idx, user_idx):
		pred = self.compute_bpr(item_idx, user_idx)
		if self.item_content_features is not None:
			pred += self.compute_vbpr(item_idx, user_idx)
		if self.item_money_features is not None:
			pred += self.compute_mvbpr(item_idx, user_idx)
		if self.item_text_features is not None:
			pred += self.compute_mtvbpr(item_idx, user_idx)
		if self.item_des_features is not None:
			pred += self.compute_mibpr(item_idx, user_idx)
		return pred

	def compute_bpr(self, item_idx, user_idx):
		latent_item = self.latent_items[item_idx]
		latent_user = self.latent_users[user_idx]
		item_bias = self.items_bias[item_idx]
		return latent_user.T @ latent_item + item_bias

	def compute_vbpr(self, item_idx, user_idx):
		item_features = self.item_content_features[item_idx]
		latent_user_content = self.latent_users_content[user_idx]
		latent_item_content = self.content_embedding_matrix @ item_features
		return latent_user_content.T @ latent_item_content + self.content_bias.T @ item_features
    
	def compute_mvbpr(self, item_idx, user_idx):
		item_money = self.item_money_features[item_idx]
		latent_user_money = self.latent_users_money[user_idx]
		latent_item_money = self.money_embedding_matrix @ item_money
		return latent_user_money.T @ latent_item_money + self.money_bias.T @ item_money

	def compute_mtvbpr(self, item_idx, user_idx):
		item_text = self.item_text_features[item_idx]
		latent_user_text = self.latent_users_text[user_idx]
		latent_item_text = self.text_embedding_matrix @ item_text
		return latent_user_text.T @ latent_item_text + self.text_bias.T @ item_text

	def compute_mibpr(self, item_idx, user_idx):
		item_des = self.item_des_features[item_idx]
		latent_user_des = self.latent_users_des[user_idx]
		latent_item_des = self.des_embedding_matrix @ item_des
		return latent_user_des.T @ latent_item_des + self.des_bias.T @ item_des

	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))

	def auc_score(self, ground_truth):
		"""
		AUCの計算を行うことと実際の相互作用の結果
        groud_truth:トレーニングデータXで予測した結果を繰り返したXまたはy
		"""

		ground_truth = ground_truth.tocsr()

		no_users, no_items = ground_truth.shape

		pid_array = np.arange(no_items, dtype=np.int32)

		scores = []

		for user_id, row in enumerate(ground_truth):
			true_pids = row.indices[row.data == 1]
			if len(true_pids):
				nids = np.setdiff1d(pid_array, true_pids)
				np.random.shuffle(nids)
				predictions = [self.predict(pid, user_id) for pid in true_pids] + [self.predict(nid, user_id) for nid in nids[:len(true_pids)]]

				grnd = np.zeros(2 * len(true_pids), dtype=np.int32)
				grnd[:len(true_pids)] = 1

				scores.append(roc_auc_score(grnd, predictions))

		return sum(scores) / len(scores)
