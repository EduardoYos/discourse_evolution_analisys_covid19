from datetime import datetime
import math
import time
import pickle

import numpy as np
import pandas as pd
import scipy
from statsmodels.formula.api import ols
import unidecode as unidecode

SAVE_FILES = True

topic_list = ['covid19_residuos', 'coronavirus', 'recomendacoes_pandemia', 'medidas_de_protecao', 'infos_pandemia']


def transform_input(psi, timestamp):
	ts_list = timestamp.unique()
	ts_dates = [datetime.fromtimestamp(x).date() for x in ts_list]

	xs = np.linspace(0, 1, num=ts_list.size)

	prob_list = []
	for i in range(len(psi)):
		ys = [math.pow(1 - x, psi[i][0] - 1) * math.pow(x, psi[i][1] - 1) / scipy.special.beta(psi[i][0], psi[i][1])
			  for x in xs]
		prob_list.append(ys)

	for i in range(len(prob_list)):
		prob_list[i] = np.asarray(prob_list[i])

	# names = ['topic1', 'topic2', 'topic3', 'topic4', 'topic5']
	df = pd.DataFrame.from_dict(dict(zip(topic_list, prob_list)))
	df.insert(0, 'Date', ts_dates)

	return df

def main():
	start_time = time.time()

	datapath = '../data_tot/'
	general_datapath = '../../data/'
	resultspath = '../results_tot/'
	tot_pickle_path = resultspath + 'covid_totnew.pickle'

	tot_pickle = open(tot_pickle_path, 'rb')
	par = pickle.load(tot_pickle)
	df_tot = transform_input(par['psi'], par['original_ts'])

	pol_df = pd.read_csv(general_datapath + 'politicas_mobilidade.csv')
	city_list = ['Rio Branco', 'Maceió', 'Macapá', 'Manaus', 'Salvador', 'Fortaleza', 'Vitória', 'Goiânia',
				 'São Luís', 'Cuiabá', 'Campo Grande', 'Belo Horizonte', 'Belém', 'João Pessoa', 'Curitiba',
				 'Recife', 'Teresina', 'Rio de Janeiro', 'Natal', 'Porto Alegre', 'Porto Velho', 'Boa Vista',
				 'Florianópolis', 'São Paulo', 'Aracaju', 'Palmas']

	city_list_normalized = [unidecode.unidecode(x.lower()) for x in city_list]
	pol_df = pol_df[pol_df['City'].isin(city_list_normalized)]
	pol_df = pol_df[(pol_df['Date'] >= '2020-04-01') & (pol_df['Date'] < '2021-01-01')].reset_index(drop=True)
	pol_df = pol_df[['State', 'City', 'Date', 'avg_mobility_7rolling', 'new_confirmed_7rolling']]
	pol_df = pol_df.groupby('Date').mean().reset_index()
	pol_df['Date'] = pd.to_datetime(pol_df['Date'])
	pol_df['Date'] = pd.to_datetime(pol_df['Date']).dt.date
	df_final = pd.merge(df_tot, pol_df, how="inner", on="Date")

	model = ols("avg_mobility_7rolling ~ covid19_residuos + coronavirus + recomendacoes_pandemia + medidas_de_protecao", data=df_final)
	response = model.fit()
	if SAVE_FILES:
		with open(resultspath + 'linear_regression_results_tot.txt', 'w', encoding='utf-8') as f:
			f.write(response.summary().as_text())
			f.write('\n')
		f.close()
	print(response.summary())

	# nova_pessoa = pd.DataFrame([{'RA': 236782, 'Height': 140}])
	# response.predict(nova_pessoa)


	print("\nExecution time: {} seconds or {} minutes"
		  .format((time.time() - start_time), (time.time() - start_time)/60))

if __name__ == "__main__":
    main()