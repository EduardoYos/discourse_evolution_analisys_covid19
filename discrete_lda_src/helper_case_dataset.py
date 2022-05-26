# script to get covid-19 cases information and treat them to generate the graph
# with topics information

# input file: "caso.csv" -> file with raw cases data
# output file: "cases_treated.csv" -> file used in graph generator script
import time
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

def adjust_number_per_month(df):
    month = relativedelta(months=+1)
    df = df.reset_index(drop=True)

    # iterate over rows (vertically)
    for index_row, row in df.iterrows():
        past_date = row['date'] - month
        past_year_month = past_date.to_period('M')
        city_df = df[row['city'] == df['city']]
        cases = city_df['confirmed'][city_df['year-month'] == past_year_month]
        if(not cases.empty):
            interval_c = row['confirmed'] - cases.tolist()[0]
            if(interval_c >= 0):
                df.loc[index_row, 'confirmed'] = interval_c
        deaths = city_df['deaths'][city_df['year-month'] == past_year_month]
        if (not deaths.empty):
            interval_d = row['deaths'] - deaths.tolist()[0]
            if (interval_d >= 0):
                df.loc[index_row, 'deaths'] = interval_d

    return df

def main():
    start_time = time.time()

    path = '../data/'
    df_cases = pd.read_csv(path + 'caso.csv')
    df_cases = df_cases[['date', 'state', 'city', 'confirmed', 'deaths']]
    # df_cases_full = pd.read_csv(path + 'caso_full.csv')
    # df_city_halls = pd.read_csv(path + 'prefeituras.csv')
    # df_city_halls = df_city_halls.drop_duplicates('Page Name')
    print(df_cases.shape[0])
    # print(df_cases_full.shape[0])

    city_list = ['Rio Branco', 'Maceió', 'Macapá', 'Manaus', 'Salvador', 'Fortaleza', 'Vitória', 'Goiânia',
                 'São Luís', 'Cuiabá', 'Campo Grande', 'Belo Horizonte', 'Belém', 'João Pessoa', 'Curitiba',
                 'Recife', 'Teresina', 'Rio de Janeiro', 'Natal', 'Porto Alegre', 'Porto Velho', 'Boa Vista',
                 'Florianópolis', 'São Paulo', 'Aracaju', 'Palmas']

    df_cases = df_cases[df_cases['city'].isin(city_list)]
    df_cases['date'] = pd.to_datetime(df_cases['date'])
    df_cases['year-month'] = df_cases['date'].dt.to_period('M')
    df_cases = df_cases[df_cases['year-month'] <= '2021-04']
    df_cases = df_cases.drop_duplicates(['city', 'year-month'], keep="first")
    df_cases = adjust_number_per_month(df_cases)

    #df_exe = df_cases[['city', 'confirmed']].groupby('city').sum().reset_index

    # df_cases2 = df_cases.drop_duplicates('city', keep="last")
    print(df_cases.shape[0])

    df_cases.to_csv(path + 'cases_treated.csv', index=False)

    print("\nExecution time: %s seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()