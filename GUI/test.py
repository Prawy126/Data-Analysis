from Dane.Dane import *

retail_df = wczytaj_dane_sklepu(wyswietlaj_informacje=True)
if retail_df is not None:
    print(retail_df.describe())

math_df = wczytaj_dane_szkolne(przedmiot="math", wyswietlaj_informacje=True)
por_df = wczytaj_dane_szkolne(przedmiot="por", wyswietlaj_informacje=True)

if math_df is not None and por_df is not None:
    math_df['przedmiot'] = 'mat'
    por_df['przedmiot'] = 'por'

    combined_students = pd.concat([math_df, por_df])
    print(combined_students.groupby('przedmiot')['G3'].mean())