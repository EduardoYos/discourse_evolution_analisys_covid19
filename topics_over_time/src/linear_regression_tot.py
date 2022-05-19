import time
# from nltk.corpus import stopwords
# import pyLDAvis
# from tot import TopicsOverTime
# import numpy as np
import pickle
import pandas as pd
from statsmodels.formula.api import ols

SAVE_FILES = True

def main():
	start_time = time.time()

	datapath = '../data_tot/'
	general_datapath = '../../data/'
	resultspath = '../results_tot/'
	tot_pickle_path = resultspath + 'covid_totnew.pickle'

	tot_pickle = open(tot_pickle_path, 'rb')
	par = pickle.load(tot_pickle)

	pol_df = pd.read_csv(general_datapath + 'politicas_mobilidade.csv')
	pol_df = pol_df[(pol_df['Date'] >= '2020-04-01') & (pol_df['Date'] < '2021-01-01')].reset_index(drop=True)
	pol_df = pol_df[['State', 'City', 'Date', 'avg_mobility_7rolling', 'new_confirmed_7rolling']]

	model = ols("avg_mobility_7rolling ~ new_confirmed_7rolling", data=pol_df)
	response = model.fit()
	if SAVE_FILES:
		with open(resultspath + 'linear_regression_results.txt', 'w', encoding='utf-8') as f:
			f.write(response.summary().as_text())
			f.write('\n')
		f.close()
	print(response.summary())

	# nova_pessoa = pd.DataFrame([{'RA': 236782, 'Height': 140}])
	# response.predict(nova_pessoa)

	# stop_words = stopwords.words('portuguese')  # removing stop words
	# stop_words.extend((list(df.city.unique())))
	# stop_words.extend(['pracegover', 'macapa', 'acre', 'natal', 'belém',
	# 				   'curitiba', 'imagem', 'texto', 'fortalezar', 'foto', 'belémcontracoronavírus'])
	#
	# tot_topic_vectors_path = resultspath + 'covid_tot_topic_vectors.csv'
	# tot_topic_mixtures_path = resultspath + 'covid_tot_topic_mixtures.csv'
	# tot_topic_shapes_path = resultspath + 'covid_tot_topic_shapes.csv'
	# tot_pickle_path = resultspath + 'covid_totnew.pickle'
	#
	# tot = TopicsOverTime()
	# documents, original_ts, timestamps, dictionary, doc_length, word_freq_list = tot.GetCovidCorpusAndDictionary(df['Message'], df['Timestamp'], stop_words)
	# par = tot.InitializeParameters(documents, original_ts, timestamps, dictionary)
	# theta, phi, psi = tot.TopicsOverTimeGibbsSampling(par)
	#
	# if SAVE_LDAVIS:
	# 	model_data = {'topic_term_dists': phi,
	# 			'doc_topic_dists': theta,
	# 			'doc_lengths': doc_length,
	# 			'vocab': dictionary,
	# 			'term_frequency': word_freq_list}
	# 	model_vis_data = pyLDAvis.prepare(mds='tsne', **model_data)
	# 	pyLDAvis.save_html(model_vis_data, '../results_tot/lda_result_tot.html')
	#
	# np.savetxt(tot_topic_vectors_path, phi, delimiter=',')
	# np.savetxt(tot_topic_mixtures_path, theta, delimiter=',')
	# np.savetxt(tot_topic_shapes_path, psi, delimiter=',')
	# tot_pickle = open(tot_pickle_path, 'wb')
	# pickle.dump(par, tot_pickle)
	# tot_pickle.close()

	print("\nExecution time: {} seconds or {} minutes"
		  .format((time.time() - start_time), (time.time() - start_time)/60))

if __name__ == "__main__":
    main()