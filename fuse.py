import math
import random
import pickle
import cProfile
import numpy as np
from collections import defaultdict
from sklearn.linear_model import Lasso, LinearRegression as LR, LogisticRegression as LogR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error as mse, roc_auc_score
from sklearn.datasets import load_digits, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

class RAVE:

    def __init__(self):

	self.gRAVE_reward = defaultdict(lambda: 0)
	self.gRAVE_visit = defaultdict(lambda: 0)

	self.lRAVE_reward = defaultdict(lambda: 0)
	self.lRAVE_visit = defaultdict(lambda: 0)

	self.g_keys = self.gRAVE_reward.keys()
	self.l_keys = self.lRAVE_reward.keys()

    def update_g(self, node, reward):

	for f in node.features:

	    self.gRAVE_reward[f] += reward
	    self.gRAVE_visit[f] += 1

	self.g_keys = self.gRAVE_reward.keys()

    def update_l(self, node, reward):
	key = node.tuple_features #tuple(sorted(node.features))
	self.lRAVE_reward[key] += reward
	self.lRAVE_visit[key] += 1

	self.l_keys = self.lRAVE_reward.keys()

class Node:

	def __init__(self, feature, features, dim, max_dim, terminal=True):

		self.feature = feature
		self.features = features
		#self.tuple_features = tuple(sorted(features))
		#self.tuple_features = frozenset(features)
		self.all_features = range(dim)
		self.remain_features = list(set(self.all_features) - set(self.features))

		self.visited = 0
		self.reward = 0
		self.reward_list = []
		self.var_reward = np.var(self.reward_list)
		self.children = []

		self.max_dim = max_dim
		self.terminal = None #terminal

	def is_expandable(self, total_visit = 0):

		#Return False if all possible childs are visited
		#Return True if not all possible childs are visited

		#if int(math.sqrt(total_visit + 1)) - int(math.sqrt(total_visit)) == 0:
		#    return False
		if int(math.sqrt(self.visited + 1)) - int(math.sqrt(self.visited)) == 0:
		    return False
		
		#if len(self.children) > 200:
		#    return False

		if len(self.remain_features) == 0:
			return False
		else:
			return True

	def is_terminal(self, total_visit):

		if self.terminal == None:
		    if len(self.features) < self.max_dim and len(self.remain_features) > 0 and random.random() < (1-1e-3) ** len(self.features):
			self.terminal = False
		    else:
			#print "Terminal Depth:%d" % len(self.features)
			self.terminal = True

		return self.terminal

		if len(self.children) == 0 and self.is_expandable(total_visit) == False:
		    return self.terminal

		if len(self.features) < self.max_dim and len(self.remain_features) > 0 and random.random() < (1-1e-2) ** len(self.features):
			self.terminal = False

		return self.terminal

	def ucb(self, total_visit, rave, beta, alpha):

		#ubc_value = self.reward * 1.0 / self.visited + math.sqrt(0.1 * math.log(total_visit) / self.visited)
		#tune_value = min(0.25, self.var_reward + math.sqrt(2.0 * math.log(total_visit) / self.visited))
		#ubc_value = self.reward * 1.0 / self.visited + math.sqrt(0.1 * math.log(total_visit) / self.visited) * tune_value

		## Considering the RAVE
		rave_value = 0#self.frave(None, rave, beta)
		tune_value = min(0.25, self.var_reward + math.sqrt(2.0 * math.log(total_visit) / self.visited))
		ubc_value = (1-alpha) * self.reward * 1.0 / self.visited + alpha * rave_value + math.sqrt(0.1 * math.log(total_visit) / self.visited) * tune_value
		
		return ubc_value

	def frave(self, feature, rave, beta):

		return 0

		if feature not in rave.l_keys:
		    lrave = 0
		else:
		    lrave = self.lRAVE_reward[f] * 1.0 / self.lRAVE_visit[f]

		#key = tuple(sorted(self.features + [feature]))
		if feature != None:
		    key = self.tuple_features | set([feature]) #tuple(sorted(self.features + [feature]))
		else:
		    key = self.tuple_features
		if key not in rave.g_keys:
		    grave = 0
		else:
		    grave = self.gRAVE_reward[key] * 1.0 / self.gRAVE_visit[key]

		return beta * grave + (1-beta) * lrave

	def update(self, reward):

		self.visited += 1
		self.reward += reward
		self.reward_list.append(reward)
		self.var_reward = np.var(self.reward_list)

class MCTS:

	def __init__(self, x, y, max_dim, max_simulation, tst_x=None, tst_y=None):

		self.x = x
		self.y = y

		self.tst_x = tst_x
		self.tst_y = tst_y

		self.dim = x.shape[1]
		self.max_dim = max_dim
		self.max_simulation = max_simulation

		self.nodes = []

		self.rave = RAVE()

		self.root = Node(-1, [], self.dim, self.max_dim, terminal=False)

	def best_feature(self):

		assert len(self.root.children) > 0

		def best_child(node):
			if node.terminal:
				return node

			best_child_nodes = [best_child(child_node) for child_node in node.children]
			best_child_id = np.argmax([child_node.visited for child_node in best_child_nodes])
			best_child_node = best_child_nodes[best_child_id]

			return best_child_node 
		return best_child(self.root)

	def fit(self):

		for i in range(self.max_simulation):
			#print i
			reward = self.select_node(self.root, self.root.visited)
			self.root.update(reward)

			if i % 5000 == 0:

				best_feature = self.best_feature()
				print "=" * 100
				print "Iter:%d, Max Visited:%d" % (i, best_feature.visited)
				
				lr = LogR(fit_intercept=True, penalty="l2")

				array_features = np.array(best_feature.features)
				lr.fit(self.x[:, array_features], self.y)
				print "#Features:%d, Train AUC:%f, Test AUC:%f" % (
					len(best_feature.features),
					roc_auc_score(self.y, lr.predict_proba(self.x[:, array_features])[:, 1]),
					roc_auc_score(self.tst_y, lr.predict_proba(self.tst_x[:, array_features])[:, 1])
					)

				#print self.rave.gRAVE_reward

	def select_node(self, node, parent_visit):

		if node.is_terminal(parent_visit):

			x_train, x_vd, y_train, y_vd = train_test_split(self.x[:, np.array(node.features)], self.y, test_size=0.5)
			
			#Regression
			# lr = LR(fit_intercept=False)
			# lr.fit(x_train, y_train)
			# reward = 1 - math.sqrt(mse(y_vd, lr.predict(x_vd)))

			#Classification
			#model = LogR(fit_intercept=True, penalty="l2")
			#model.fit(x_train, y_train)
			model = KNN(n_neighbors=5)
			model.fit(x_train, y_train)
			
			reward = roc_auc_score(y_vd, model.predict_proba(x_vd)[:, 1])

			#print reward

			node.update(reward)

			#self.rave.update_g(node, reward)
			#self.rave.update_l(node, reward)

			return reward

		if node.is_expandable(parent_visit):
			#Random select one node from the remain features
			rd_feature_id = random.randint(0, len(node.remain_features) - 1)
			rd_feature = node.remain_features.pop(rd_feature_id)

			#select one node according to RAVE
			#rd_feature_id = np.argmax([node.frave(feature, self.rave, 0.5) for feature in node.remain_features[0:1000]])	
			#rd_feature = node.remain_features.pop(rd_feature_id)

			child_node = Node(rd_feature, node.features + [rd_feature], self.dim, self.max_dim)
			node.children.append(child_node)

			reward = self.select_node(child_node, node.visited)
			node.update(reward)
			#self.rave.update_l(node, reward)

		else:
			max_ucb_id = np.argmax([child_node.ucb(node.visited, self.rave, 0.5, 0.0) for child_node in node.children])
			max_ucb_node = node.children[max_ucb_id]

			reward = self.select_node(max_ucb_node, node.visited)
			node.update(reward)
			#self.rave.update_l(node, reward)

		return reward


if __name__ == "__main__":

	#digits = load_digits()
	# print digits.data.shape
	# print digits.target

	#source_data = digits.data[np.logical_or(digits.target == 4, digits.target == 7), :]
	#source_label = digits.target[np.logical_or(digits.target == 4, digits.target == 7)]
	#source_label[source_label == 4] = 1
	#source_label[source_label == 7] = 0

	#target_data = digits.data[np.logical_or(digits.target == 4, digits.target == 9), :]
	#target_label = digits.target[np.logical_or(digits.target == 4, digits.target == 9)]
	#target_label[target_label == 4] = 1
	#target_label[target_label == 9] = 0
	
	newsgroups_train = fetch_20newsgroups(subset="train", categories=['rec.autos', 'rec.motorcycles', "sci.crypt", "sci.electronics"])

	vectorizer = TfidfVectorizer()
	data = vectorizer.fit_transform(newsgroups_train.data)
	label = newsgroups_train.target

	source_data = data[label <= 1, :]
	source_label = label[label <= 1]

	target_data = data[label > 1, :]
	target_label = label[label > 1]

	source_tr_data, source_tst_data, source_tr_label, source_tst_label = train_test_split(source_data, source_label, test_size=0.5)
	target_tr_data, target_tst_data, target_tr_label, target_tst_label = train_test_split(target_data, target_label, test_size=0.5)

	pickle.dump([source_tr_data, source_tst_data, source_tr_label, source_tst_label, target_tr_data, target_tst_data, target_tr_label, target_tst_label], open("20newsdata.pkl", "w"))
	source_tr_data, source_tst_data, source_tr_label, source_tst_label, target_tr_data, target_tst_data, target_tr_label, target_tst_label = pickle.load(open("20newsdata.pkl"))

	# source_n = 40
	# source_tr_data = source_tr_data[0:source_n, :]
	# source_tr_label = source_tr_label[0:source_n]

	#print source_tr_label

	lr = LogR(fit_intercept=True, penalty="l1")
	lr.fit(source_tr_data, source_tr_label)
	print "NNZ:%d, Train AUC:%f, Test AUC:%f" % (
		np.sum(lr.coef_ != 0),	
		roc_auc_score(source_tr_label, lr.predict_proba(source_tr_data)[:, 1]),
		roc_auc_score(source_tst_label, lr.predict_proba(source_tst_data)[:, 1])
		)

	#print lr.coef_.shape
	max_dim_id = np.argsort(np.abs(lr.coef_.flatten()))[-500:]
	#print np.abs(lr.coef_.flatten())[max_dim_id]

	source_tr_data = source_tr_data[:, max_dim_id]
	source_tst_data = source_tst_data[:, max_dim_id]
	target_tr_data = target_tr_data[:, max_dim_id]
	target_tst_data = target_tst_data[:, max_dim_id]

	#print source_data.shape
	mcts = MCTS(source_tr_data, source_tr_label, 64, 500000, source_tst_data, source_tst_label)
	cProfile.run("mcts.fit()")
	#mcts.fit()

	# print source_label.shape
	# print target_label.shape

	# N = 200
	# dim = 50

	# #x = np.random.normal(loc=0, scale=1, size=(N, dim)) # N * dim
	# x = np.random.rand(N, dim)
	# w = np.random.rand(dim)
	# w[5::] = 0

	# y = np.dot(x, w) + x[:, 5] * x[:, 6] + x[:, 7] * x[:, 8] * x[:, 9] + np.random.normal(loc=0, scale=0.1, size=N)

	# mcts = MCTS(x,y,20,500000)
	# mcts.fit()

	# ls = Lasso(alpha=0.001)
	# ls.fit(x, y)
	# print np.abs(ls.coef_)[np.argsort(np.abs(ls.coef_))]
	# print np.arange(dim)[np.argsort(np.abs(ls.coef_))[::-1]]
