# 📚 Spis treści funkcji – katalog `Backend`

## Czyszczenie.py
- **ekstrakcja_podtablicy(df, rows, cols, mode, wyswietlaj_informacje)**  
  _Ekstrakcja podtablicy przez usuwanie lub zachowywanie wskazanych wierszy/kolumn._

---

## Duplikaty.py
- **usun_duplikaty(df, kolumny, tryb, wyswietlaj_info)**  
  _Usuwa powtarzające się wiersze z DataFrame'a._

---

## Kodowanie.py
- **jedno_gorace_kodowanie(df, kolumny, usun_pierwsza, wyswietl_informacje)**  
  _Kodowanie One-Hot dla wybranych kolumn kategorycznych._
- **binarne_kodowanie(df, kolumny, wyswietlaj_informacje)**  
  _Kodowanie binarne (Binary Encoding) dla kolumn kategorycznych._
- **kodowanie_docelowe(df, kolumny, target, smoothing, wyswietlaj_informacje)**  
  _Kodowanie Target Encoding dla kolumn kategorycznych._

---

## Korelacje.py
- **oblicz_korelacje_pearsona(sciezka_pliku, separator, kolumny_daty, format_daty, wymagane_kolumny, wyswietlaj_informacje)**  
  _Oblicza macierz korelacji Pearsona dla numerycznych kolumn w pliku CSV._
- **oblicz_korelacje_spearmana(sciezka_pliku, separator, kolumny_daty, format_daty, wymagane_kolumny, wyswietlaj_informacje)**  
  _Oblicza macierz korelacji Spearmana._

---

## Skalowanie.py
- **minmax_scaler(df, kolumny, wyswietlaj_informacje)**  
  _Skaluje wybrane kolumny numeryczne do zakresu [0, 1] metodą MinMax._
- **standard_scaler(df, kolumny, wyswietlaj_informacje)**  
  _Standaryzuje wybrane kolumny numeryczne do średniej 0 i wariancji 1._

---

## Statystyka.py
- **oblicz_statystyki_nie_numeryczne(df)**  
  _Oblicza statystyki dla kolumn nie-numerycznych._
- **znajdz_kolumny_numeryczne(df)**  
  _Znajduje wszystkie kolumny numeryczne._
- **wydobadz_wartosci_numeryczne(df, wybrane_kolumny)**  
  _Wydobywa wartości numeryczne z kolumn._
- **analizuj_dane_numeryczne(sciezka_pliku, separator, wybrane_kolumny)**  
  _Wczytuje dane, wyodrębnia numeryczne i oblicza podstawowe statystyki._
- **oblicz_statystyki(wartosci_numeryczne)**  
  _Oblicza podstawowe statystyki dla każdej kolumny numerycznej._
- **srednia_wszystkich_wartosci_numerycznych(wartosci_numeryczne)**  
  _Oblicza średnią ze wszystkich wartości numerycznych._

---

## Uzupelniane.py
- **uzupelnij_braki(df, metoda, wartosc_stala, reguly, wyswietlaj_info)**  
  _Wypełnia brakujące wartości różnymi strategiami._
- **usun_braki(df, os_wiersze_kolumny, liczba_min_niepustych, wyswietlaj_info)**  
  _Usuwa wiersze lub kolumny zawierające brakujące dane._

---

## Wartosci.py
- **zamien_wartosci(df, kolumna, stara_wartosc, nowa_wartosc, reguly, wyswietlaj_informacje)**  
  _Zastępuje wartości w DataFrame ręcznie lub automatycznie wg reguł._

---

## Wykresy.py
- **rysuj_wykres(df, typ_wykresu, ...)**  
  _Uniwersalna funkcja do rysowania różnych typów wykresów (scatter, bar, line, heatmap, pie itd.)._

---

> Dla pełnej listy plików i najnowszej zawartości odwiedź katalog [Backend](https://github.com/Prawy126/HurtownieDanych/tree/main/Backend).
