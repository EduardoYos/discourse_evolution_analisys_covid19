import time
from nltk.corpus import stopwords
import pyLDAvis
from tot import TopicsOverTime
import numpy as np
import pickle
import pandas as pd

SAVE_LDAVIS = True

def main():
	start_time = time.time()

	datapath = '../data_tot/'
	resultspath = '../results_tot/'
	df = pd.read_csv(datapath + 'covid_dataset_treated.csv')
	df = df[(df['Date'] >= '2020-04-01') & (df['Date'] < '2021-01-01')].reset_index(drop=True)

	stop_words = stopwords.words('portuguese')  # removing stop words
	stop_words.extend((list(df.city.unique())))
	stop_words.extend(['pracegover', 'macapa', 'acre', 'natal', 'belém',
					   'curitiba', 'imagem', 'texto', 'fortalezar', 'foto', 'belémcontracoronavírus'])

	tot_topic_vectors_path = resultspath + 'covid_tot_topic_vectors.csv'
	tot_topic_mixtures_path = resultspath + 'covid_tot_topic_mixtures.csv'
	tot_topic_shapes_path = resultspath + 'covid_tot_topic_shapes.csv'
	tot_pickle_path = resultspath + 'covid_tot_last.pickle'

	tot = TopicsOverTime()
	documents, original_ts, timestamps, dictionary, doc_length, word_freq_list = tot.GetCovidCorpusAndDictionary(df['Message'], df['Timestamp'], stop_words)
	par = tot.InitializeParameters(documents, original_ts, timestamps, dictionary)
	theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)

	if SAVE_LDAVIS:
		model_data = {'topic_term_dists': phi,
				'doc_topic_dists': theta,
				'doc_lengths': doc_length,
				'vocab': dictionary,
				'term_frequency': word_freq_list}
		model_vis_data = pyLDAvis.prepare(mds='tsne', **model_data)
		pyLDAvis.save_html(model_vis_data, '../results_tot/lda_result_tot.html')

		np.savetxt(tot_topic_vectors_path, phi, delimiter=',')
		np.savetxt(tot_topic_mixtures_path, theta, delimiter=',')
		np.savetxt(tot_topic_shapes_path, psi, delimiter=',')
		tot_pickle = open(tot_pickle_path, 'wb')
		pickle.dump(par, tot_pickle)
		tot_pickle.close()

	print("\nExecution time: {} seconds or {} minutes"
		  .format((time.time() - start_time), (time.time() - start_time)/60))

if __name__ == "__main__":
    main()