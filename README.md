# Analiza Online Retail II i Student Performance

Aplikacja do analizy statystycznej, klasteryzacji, klasyfikacji oraz przetwarzania wstępnego danych z wykorzystaniem zbiorów **Online Retail II** i **Student Performance**.

---

## Spis treści
1. [Opis projektu](#opis-projektu)
2. [Funkcjonalności](#funkcjonalności)
3. [Wykorzystane technologie](#wykorzystane-technologie)
4. [Instalacja i wymagania](#instalacja-i-wymagania)
5. [Uruchomienie aplikacji](#uruchomienie-aplikacji)
6. [Interfejs użytkownika (GUI)](#interfejs-użytkownika-gui)
7. [Eksperymenty i wyniki](#eksperymenty-i-wyniki)
8. [Autorzy](#autorzy)
9. [Literatura](#literatura)

---

## Opis projektu
Celem projektu jest stworzenie narzędzia umożliwiającego:
- **Analizę statystyczną** (min, max, odchylenie standardowe, mediana, moda).
- **Klasteryzację** (K-Means) i **klasyfikację** (Random Forest, ID3).
- **Preprocessing danych** (usuwanie brakujących wartości, kodowanie, skalowanie).
- **Wizualizację** (wykresy słupkowe, liniowe, punktowe, kołowe).
- Pracę na dwóch zbiorach danych: **Online Retail II** (transakcje e-commerce) i **Student Performance** (wyniki uczniów).

---

## Funkcjonalności
| Funkcjonalność                  | Opis                                                                 |
|---------------------------------|---------------------------------------------------------------------|
| **Wczytywanie danych**          | Obsługa plików CSV z walidacją błędów.                              |
| **Statystyki**                  | Obliczanie miar statystycznych dla danych numerycznych i kategorycznych. |
| **Korelacje**                   | Metody Pearsona i Spearmana.                                        |
| **Modyfikacja danych**          | Usuwanie kolumn/wierszy, zastępowanie wartości, skalowanie (MinMax, Standard). |
| **Kodowanie**                   | One-Hot Encoding, Binary Encoding, Target Encoding.                |
| **Wykresy**                     | 4 typy: słupkowy, liniowy, punktowy, kołowy.                       |
| **Algorytmy ML**                | Klasyfikacja (Random Forest, ID3), klasteryzacja (K-Means).        |

---

## Wykorzystane technologie
- **Język**: Python 3.12
- **Biblioteki**: 
  - `pandas`, `numpy` – przetwarzanie danych.
  - `scikit-learn` – algorytmy ML i preprocessing.
  - `matplotlib`, `seaborn` – wizualizacja.
  - `tkinter` – interfejs graficzny.
- **Narzędzia**: Jupyter Notebook (eksperymenty), Git (wersjonowanie).

---

## Instalacja i wymagania
### Wymagania sprzętowe/programowe:
- System: Windows/Linux/macOS.
- RAM: min. 4 GB (dla dużych zbiorów danych zalecane 8 GB).
- Python 3.12+.

### Instalacja:
1) Sklonuj repozytorium:

```bash
git clone https://github.com/twoj_nick/projekt.git
```

---

## Uruchomienie aplikacji
Uruchom plik główny:
   
2) Zainstaluj zależności:

```bash
pip install -r requirements.txt
```

---

## Uruchomienie aplikacji
1) **Uruchom plik główny:**

```bash
python main.py
```
2) **Krok po kroku:**
- Wybierz zbiór danych (CSV).
- Wykonaj preprocessing (np. usuwanie brakujących wartości).
- Wybierz funkcjonalność (statystyki, wykresy, algorytmy ML).
- Zapisz wyniki lub eksportuj wykresy.

---

## Interfejs użytkownika (GUI)

_Aktualnie trwają prace_

---

## Eksperymenty i wyniki
1) Zbiory danych:
- Online Retail II:
  - Opis: 1,067,371 transakcji e-commerce.
  - Wyniki: Klasteryzacja produktów według sprzedaży (K-Means), wykrycie sezonowości zakupów.
- Student Performance:
  - Opis: 649 uczniów, 30 cech.
  - Wyniki: Klasyfikacja wyników z matematyki (Random Forest – dokładność 85%).
- Wnioski:
  - K-Means skutecznie grupuje produkty o podobnej sprzedaży.
  - Random Forest osiąga lepsze wyniki niż ID3 w przewidywaniu ocen.

---

## Autorzy

**Jakub Opar** – frontend (GUI, wizualizacja).

**Michał Pilecki** – backend (algorytmy ML, statystyki).

---

## Literatura

- Zbiór Online Retail II: UCI Machine Learning Repository.
- Zbiór Student Performance: UCI Machine Learning Repository.
- Dokumentacja bibliotek: pandas, scikit-learn, matplotlib.
