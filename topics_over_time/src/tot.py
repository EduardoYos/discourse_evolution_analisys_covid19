from copy import deepcopy

import nltk
# nltk.download('stopwords')
import fileinput
import random
import scipy.special
import math
import numpy as np
import scipy.stats
import pickle
from math import log
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords as stpwrd


class TopicsOverTime:
	def GetCovidCorpusAndDictionary(self, df_documents, df_timestamps, stop_words):
		documents = []
		doc_length = []
		word_list = []
		timestamps = df_timestamps
		dictionary = set()
		# stopwords = stpwrd.words('english')

		# stop_words = [word.lower().strip().split() for word in stop_words]
		# for line in stop_words:
		# 	line = line.lower().strip().split()
		for doc in df_documents:
			# words = [word for word in doc.lower().strip().split() if word not in stop_words
			# 		 and (3 < len(word) < 30)]
			words = word_tokenize(text=doc.lower(), language="portuguese")
			words = list(filter(lambda x: x not in string.punctuation and
								 x not in stop_words and (3 < len(x) < 30), words))
			documents.append(words)
			word_list += words
			dictionary.update(set(words))
			doc_length.append((len(doc)))
		# for timestamp in fileinput.input(timestamps_path):
		# 	num_titles = int(timestamp.strip().split()[0])
		# 	timestamp = float(timestamp.strip().split()[1])
		# 	timestamps.extend([timestamp for title in range(num_titles)])
		# for line in fileinput.input(stopwords_path):
		# 	stopwords.update(Set(line.lower().strip().split()))
		word_freq_list = nltk.FreqDist(word_list).most_common()
		original_ts = timestamps
		first_timestamp = timestamps[0]
		last_timestamp = timestamps[len(timestamps)-1]
		timestamps = [1.0*(t-first_timestamp)/(last_timestamp-first_timestamp) for t in timestamps]
		dictionary = list(dictionary)
		assert len(documents) == len(timestamps)
		return documents, original_ts, timestamps, dictionary, doc_length, word_freq_list

	def CalculateCounts(self, par):
		for d in range(par['D']):
			for i in range(par['N'][d]):
				topic_di = par['z'][d][i]		#topic in doc d at position i
				word_di = par['w'][d][i]		#word ID in doc d at position i
				par['m'][d][topic_di] += 1
				par['n'][topic_di][word_di] += 1
				par['n_sum'][topic_di] += 1

	def InitializeParameters(self, documents, original_ts, timestamps, dictionary):
		par = {}						# dictionary of all parameters
		par['dataset'] = 'covid'			# dataset name
		par['max_iterations'] = 100	# max number of iterations in gibbs sampling
		par['T'] = 5					# number of topics
		par['D'] = len(documents)
		par['V'] = len(dictionary)
		par['N'] = [len(doc) for doc in documents]
		par['alpha'] = [50.0/par['T'] for _ in range(par['T'])]
		par['beta'] = [0.1 for _ in range(par['V'])]
		par['beta_sum'] = sum(par['beta'])
		par['psi'] = [[1 for _ in range(2)] for _ in range(par['T'])]
		par['betafunc_psi'] = [scipy.special.beta( par['psi'][t][0], par['psi'][t][1] ) for t in range(par['T'])]
		par['word_id'] = {dictionary[i]: i for i in range(len(dictionary))}
		par['word_token'] = dictionary
		par['original_ts'] = original_ts
		par['z'] = [[random.randrange(0,par['T']) for _ in range(par['N'][d])] for d in range(par['D'])]
		par['t'] = [[timestamps[d] for _ in range(par['N'][d])] for d in range(par['D'])]
		par['w'] = [[par['word_id'][documents[d][i]] for i in range(par['N'][d])] for d in range(par['D'])]
		par['m'] = [[0 for t in range(par['T'])] for d in range(par['D'])]
		par['n'] = [[0 for v in range(par['V'])] for t in range(par['T'])]
		par['n_sum'] = [0 for t in range(par['T'])]
		np.set_printoptions(threshold=np.inf)
		np.seterr(divide='ignore', invalid='ignore')
		self.CalculateCounts(par)
		return par

	def GetTopicTimestamps(self, par):
		topic_timestamps = []
		for topic in range(par['T']):
			current_topic_timestamps = []
			current_topic_doc_timestamps = [[ (par['z'][d][i]==topic)*par['t'][d][i] for i in range(par['N'][d])] for d in range(par['D'])]
			for d in range(par['D']):
				current_topic_doc_timestamps[d] = filter(lambda x: x!=0, current_topic_doc_timestamps[d])
			for timestamps in current_topic_doc_timestamps:
				current_topic_timestamps.extend(timestamps)
			assert current_topic_timestamps != []
			topic_timestamps.append(current_topic_timestamps)
		return topic_timestamps

	def GetMethodOfMomentsEstimatesForPsi(self, par):
		topic_timestamps = self.GetTopicTimestamps(par)
		psi = [[1 for _ in range(2)] for _ in range(len(topic_timestamps))]
		for i in range(len(topic_timestamps)):
			current_topic_timestamps = topic_timestamps[i]
			timestamp_mean = np.mean(current_topic_timestamps)
			timestamp_var = np.var(current_topic_timestamps)
			if timestamp_var == 0:
				timestamp_var = 1e-6
			common_factor = timestamp_mean*(1-timestamp_mean)/timestamp_var - 1
			psi[i][0] = 1 + timestamp_mean*common_factor
			psi[i][1] = 1 + (1-timestamp_mean)*common_factor
		return psi

	def ComputePosteriorEstimatesOfThetaAndPhi(self, par):
		theta = deepcopy(par['m'])
		phi = deepcopy(par['n'])

		for d in range(par['D']):
			if sum(theta[d]) == 0:
				theta[d] = np.asarray([1.0/len(theta[d]) for _ in range(len(theta[d]))])
			else:
				theta[d] = np.asarray(theta[d])
				theta[d] = 1.0*theta[d]/sum(theta[d])
		theta = np.asarray(theta)

		for t in range(par['T']):
			if sum(phi[t]) == 0:
				phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
			else:
				phi[t] = np.asarray(phi[t])
				phi[t] = 1.0*phi[t]/sum(phi[t])
		phi = np.asarray(phi)

		return theta, phi

	def ComputePosteriorEstimatesOfTheta(self, par):
		theta = deepcopy(par['m'])

		for d in range(par['D']):
			if sum(theta[d]) == 0:
				theta[d] = np.asarray([1.0/len(theta[d]) for _ in range(len(theta[d]))])
			else:
				theta[d] = np.asarray(theta[d])
				theta[d] = 1.0*theta[d]/sum(theta[d])

		return np.matrix(theta)

	def ComputePosteriorEstimateOfPhi(self, par):
		phi = deepcopy(par['n'])

		for t in range(par['T']):
			if sum(phi[t]) == 0:
				phi[t] = np.asarray([1.0/len(phi[t]) for _ in range(len(phi[t]))])
			else:
				phi[t] = np.asarray(phi[t])
				phi[t] = 1.0*phi[t]/sum(phi[t])

		return np.matrix(phi)

	def TopicsOverTimeGibbsSampling(self, par):
		for iteration in range(par['max_iterations']):
			for d in range(par['D']):
				for i in range(par['N'][d]):
					word_di = par['w'][d][i]
					t_di = par['t'][d][i]

					old_topic = par['z'][d][i]
					par['m'][d][old_topic] -= 1
					par['n'][old_topic][word_di] -= 1
					par['n_sum'][old_topic] -= 1

					topic_probabilities = []
					for topic_di in range(par['T']):
						psi_di = par['psi'][topic_di]
						topic_probability = 1.0 * (par['m'][d][topic_di] + par['alpha'][topic_di])
						topic_probability *= ((1-t_di)**(psi_di[0]-1)) * ((t_di)**(psi_di[1]-1))
						topic_probability /= par['betafunc_psi'][topic_di]
						topic_probability *= (par['n'][topic_di][word_di] + par['beta'][word_di])
						topic_probability /= (par['n_sum'][topic_di] + par['beta_sum'])
						topic_probabilities.append(topic_probability)
					sum_topic_probabilities = sum(topic_probabilities)
					if sum_topic_probabilities == 0:
						topic_probabilities = [1.0/par['T'] for _ in range(par['T'])]
					else:
						topic_probabilities = [p/sum_topic_probabilities for p in topic_probabilities]
					
					new_topic = list(np.random.multinomial(1, topic_probabilities, size=1)[0]).index(1)
					par['z'][d][i] = new_topic
					par['m'][d][new_topic] += 1
					par['n'][new_topic][word_di] += 1
					par['n_sum'][new_topic] += 1

				if d%1000 == 0:
					print('Done with iteration {iteration} and document {document}'.format(iteration=iteration, document=d))
			par['psi'] = self.GetMethodOfMomentsEstimatesForPsi(par)
			par['betafunc_psi'] = [scipy.special.beta( par['psi'][t][0], par['psi'][t][1] ) for t in range(par['T'])]
		par['m'], par['n'] = self.ComputePosteriorEstimatesOfThetaAndPhi(par)
		return par['m'], par['n'], par['psi']
