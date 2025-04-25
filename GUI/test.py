# plik: analysis_script.py
from Dane.Dane import load_retail_data, load_student_math_data, load_student_por_data

# TODO: Wczytanie danych z pliku CSV zmienić aby brało ściężkę względną a nie bezwględną
# wszystko działa tylko należy zrobić tego TODO aby pracwoało się wygodniej

df = load_retail_data(verbose=True)

df = load_student_math_data(verbose=True)

df = load_student_por_data(verbose=True)