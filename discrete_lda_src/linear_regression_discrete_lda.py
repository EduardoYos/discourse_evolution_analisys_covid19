import time
import numpy as np
import pandas as pd
import unidecode as unidecode
from statsmodels.formula.api import ols

SAVE_FILES = True

def main():
	start_time = time.time()

	datapath = '../data/'
	resultspath = '../results/'

	df_doc = pd.read_csv(datapath + 'document_topic_helper.csv')
	df_doc.rename(columns={'post_created_date': 'Date'}, inplace=True)
	df_doc['Date'] = pd.to_datetime(df_doc['Date']).dt.date
	df_doc = df_doc.groupby('Date').mean().reset_index()

	pol_df = pd.read_csv(datapath + 'politicas_mobilidade.csv')
	val = pol_df.City.unique()

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
	df_final = pd.merge(df_doc, pol_df, how="inner", on="Date")
	df_final.rename(columns={'topic1': 'casos_residuos', 'topic2': 'medidas_de_protecao',
						   'topic3': 'coronavirus', 'topic4': 'saude_e_pandemia',
						   'topic5': 'informacoes_sobre_pandemia'}, inplace=True)

	model = ols("avg_mobility_7rolling ~ casos_residuos + medidas_de_protecao + coronavirus + saude_e_pandemia", data=df_final)
	response = model.fit()
	if SAVE_FILES:
		with open(resultspath + 'lin_reg_results_discrete_lda.txt', 'w', encoding='utf-8') as f:
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