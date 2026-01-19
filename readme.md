<img width="1460" height="570" alt="image" src="https://github.com/user-attachments/assets/5b045468-0218-4a31-87df-39da2393d508" /># Nazov projektu: Stock market prediction
- Veduci prace: Oliver Halas

# 📈 Stock Price Prediction with LSTM & Transformer (PyTorch)

Tento projekt sa zameriava na **predikciu cien akcií** pomocou neurónových sietí v PyTorchi. Implementované sú dva prístupy:

* **LSTM (Long Short-Term Memory)**
* **Transformer Encoder**

Modely pracujú s časovými radmi a učia sa predikovať budúcu cenu akcie na základe niekoľkých predchádzajúcich dní.

---

## 🧠 Krátky opis projektu

Projekt:

* sťahuje historické dáta cien akcií (stĺpec `Close`) z Google Drive,
* pripravuje časový rad pomocou *lookback window* (posunuté hodnoty v čase),
* normalizuje dáta pomocou `MinMaxScaler`,
* trénuje neurónovú sieť (LSTM alebo Transformer),
* vyhodnocuje model na validačných dátach,
* loguje tréning do **TensorBoard**,
* ukladá model a pomocné objekty po každej epoche.

Celý tréning je opakovaný viackrát (viac behov) pre lepšie porovnanie výsledkov.

---

## 🗂️ Použité technológie

* **Python**
* **PyTorch**
* **Pandas / NumPy**
* **scikit-learn**
* **TensorBoard**

---

## 📦 Requirements

Odporúčaná verzia Pythonu: **Python 3.9+**

Nainštaluj potrebné knižnice:

```bash
pip install torch pandas numpy scikit-learn tensorboard
```

Ak máš CUDA-kompatibilnú GPU, tréning sa automaticky presunie na GPU.

---

## 📊 Dáta

* Dáta sú načítané priamo z **Google Drive** pomocou `pandas.read_csv`
* Používa sa iba:

  * `Date`
  * `Close`
* `Date` je konvertovaný na `datetime`
* Dáta sú spracované do sekvencií dĺžky **lookback = 7**

---

## ⚙️ Spracovanie dát

1. Vytvorenie časových posunov (`Close(t-1) ... Close(t-n)`)
2. Odstránenie `NaN` hodnôt
3. Normalizácia na rozsah `(-1, 1)`
4. Rozdelenie dát:

   * **95 % tréning**
   * **5 % test**
5. Konverzia na PyTorch tensory
6. Vytvorenie vlastného `Dataset` a `DataLoader`

---

## 🧩 Modely

### 🔁 LSTM

* Viacvrstvová LSTM sieť
* Vhodná na sekvenčné dáta
* Výstupom je predikcia ďalšej hodnoty ceny

### 🔀 Transformer

* Transformer Encoder s multi-head attention
* Lineárna projekcia vstupu
* Rýchlejší a flexibilnejší než LSTM pri dlhších sekvenciách

Model si vieš prepínať tu:

```python
model = TransformerModel(1, 4, 1)
# model = LSTM(1, 4, 1)
```

---

## 🏋️ Tréning

* Loss funkcia: **MSELoss**
* Optimizer: **Adam**
* Learning rate: `0.001`
* Epochy: `100`
* Batch size: `16`

Po každej epoche:

* prebehne validácia,
* výsledky sa zapíšu do TensorBoard,
* model sa uloží pomocou funkcie `save()`.

---

## 📈 TensorBoard

Spustenie TensorBoard:

```bash
tensorboard --logdir runs
```

Uvidíš:

* tréningový loss
* validačný loss
* priebeh učenia pre jednotlivé behy

---

## 💾 Ukladanie modelu

Model, scaler a ďalšie potrebné objekty sa ukladajú pomocou funkcie:

```python
save(model, X_train, device, lookback, scaler, writer, X_test)
```

To umožňuje neskoršie:

* načítanie modelu,
* spätnú transformáciu hodnôt,
* testovanie na nových dátach.

---

## 🚀 Možné rozšírenia

* Predikcia viac dní dopredu
* Pridanie ďalších vstupných feature (Open, High, Volume)
* Porovnanie viacerých Transformer konfigurácií
* Vizualizácia predikcií vs. realita

---

## ✍️ Autor

Projekt vytvorený ako experiment s časovými radmi a modernými neurónovými sieťami v PyTorchi.

---

Ak chceš, viem ti:

* README ešte **viac zjednodušiť** (napr. pre odovzdanie do školy),
* alebo spraviť **anglickú verziu**,
* prípadne ho upraviť presne podľa **GitHub štýlu**, ak mi povieš účel projektu.
