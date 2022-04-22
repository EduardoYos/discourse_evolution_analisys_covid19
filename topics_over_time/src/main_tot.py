import time

from nltk.corpus import stopwords

from tot import TopicsOverTime
import numpy as np
import pickle
import pandas as pd


def main():
	start_time = time.time()

	datapath = '../data_tot/'
	resultspath = '../results_tot/'
	df = pd.read_csv(datapath + 'covid_dataset_treated.csv')

	stop_words = stopwords.words('portuguese')  # removing stop words
	stop_words.extend((list(df.city.unique())))
	stop_words.extend(['pracegover', 'macapa', 'acre', 'natal', 'belém',
					   'curitiba', 'imagem', 'texto', 'fortalezar', 'foto', 'belémcontracoronavírus'])

	tot_topic_vectors_path = resultspath + 'covid_tot_topic_vectors.csv'
	tot_topic_mixtures_path = resultspath + 'covid_tot_topic_mixtures.csv'
	tot_topic_shapes_path = resultspath + 'covid_tot_topic_shapes.csv'
	tot_pickle_path = resultspath + 'covid_tot.pickle_new'

	tot = TopicsOverTime()
	documents, timestamps, dictionary = tot.GetCovidCorpusAndDictionary(df['Message'], df['Timestamp'], stop_words)
	par = tot.InitializeParameters(documents, timestamps, dictionary)
	theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)
	np.savetxt(tot_topic_vectors_path, phi, delimiter=',')
	np.savetxt(tot_topic_mixtures_path, theta, delimiter=',')
	np.savetxt(tot_topic_shapes_path, psi, delimiter=',')
	tot_pickle = open(tot_pickle_path, 'wb')
	pickle.dump(par, tot_pickle)
	tot_pickle.close()

	print("\nExecution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()