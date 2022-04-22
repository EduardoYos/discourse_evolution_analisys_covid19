import time
import pandas as pd
import numpy as np
import warnings
import datetime
import re


def set_df_date(df):
    df['day'] = pd.to_datetime(df['Post Created Date'])
    df['ano_mes'] = df['day'].dt.to_period('M')
    df['week'] = df['day'].dt.week
    df['month'] = df['day'].dt.month
    df['year'] = df['day'].dt.year
    return df

def get_covid_posts(df, covid_words):
    len_total = len(df)
    df = df[df['Message'].str.contains("|".join(covid_words), case=False, na=False)]
    len_covid = len_total - len(df)
    print('{} posts foram removidos, {:.2f}% do total'.format(len_covid, (len_covid/len_total)*100))
    return df

def main():
    start_time = time.time()
    warnings.filterwarnings("ignore")

    # region New DFs
    initial_data_path = '../../data/'
    new_data_path = '../data_tot/'

    df_city_halls = pd.read_csv(initial_data_path + 'prefeituras.csv')
    df_mayor_city = pd.read_csv(initial_data_path + 'prefeito-cidade.csv')
    df_20 = pd.read_csv(initial_data_path + 'prefeitos2020.csv')
    df_21 = pd.read_csv(initial_data_path + 'prefeitos2021.csv')
    df_city_halls['city'] = df_city_halls['Page Name'].apply(
        lambda x: x.replace('do ', 'de ').partition(' de ')[2].lower())

    df_20['city'] = df_20['Page Name'].apply(lambda x: df_mayor_city[df_mayor_city['mayor'] == x].city.values[0])
    df_21['city'] = df_21['Page Name'].apply(lambda x: df_mayor_city[df_mayor_city['mayor'] == x].city.values[0])

    df_mayors = pd.concat([df_20, df_21])
    df_last = pd.concat([df_city_halls[['Message', 'city', 'Post Created Date']],
                          df_mayors[['Message', 'city', 'Post Created Date']]])

    # df_last = set_df_date(df_last)
    # endregion

    list_covid_words = ['comorbidade', 'ivermectina', 'tamiflu', 'azitromicina', 'lockdown', 'distanciamento social',
                   'hidroxicloroquina', 'covid19', 'isolamento', 'cloroquina', 'quarentena', 'covid_19', 'coronavirus',
                   'pandemia', 'corona', 'covid']
    df_last = get_covid_posts(df_last, list_covid_words)

    # removing unusual characters
    df_last['Message'] = df_last['Message'].str.replace('[^\w\s.]', '', flags=re.UNICODE)
    # removing http/https links
    df_last['Message'] = df_last['Message'].str.replace(r'(?:http|https)\:\/\/[a-zA-Z0-9\.\/\?\:@\-_=#]+', '')

    # df_last = df_last.reset_index(drop=True)
    df_last = df_last.rename(columns={'Post Created Date': 'Date'})
    df_last['Timestamp'] = df_last['Date'].apply(
        lambda x: datetime.datetime.strptime(x, "%Y-%m-%d").timestamp())
    # df_last['Date'] = pd.to_datetime(df_last['Date'])
    df_last = df_last.sort_values(by="Date", ascending=True).reset_index(drop=True)
    # df_last = df_last[['Message', 'Date', 'Timestamp']]

    df_last.to_csv(new_data_path + 'covid_dataset_treated.csv', index=False)

    print('Total de posts: {}'.format(df_last.shape[0]))

    print("\nExecution time: %s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    main()
