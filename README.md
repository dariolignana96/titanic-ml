
# 🚢 Titanic - Machine Learning from Disaster

Analisi predittiva e classificazione della sopravvivenza dei passeggeri del Titanic tramite tecniche di Machine Learning.

## 📋 Descrizione

Questo progetto affronta la sfida Kaggle **"Titanic: Machine Learning from Disaster"**, con l'obiettivo di costruire un modello predittivo della probabilità di sopravvivenza dei passeggeri, utilizzando feature ingegnerizzate e un approccio data science completo.

**Le fasi principali:**
- Esplorazione e pulizia del dataset
- Analisi statistica e pattern di sopravvivenza
- Preprocessing, feature engineering e scaling
- Costruzione e validazione di modelli ML (Random Forest)
- Visualizzazione e interpretazione dei risultati

---

## 📂 Struttura del repository

```
titanic-ml/
├── data/
│   └── train.csv           # Dataset di training (scaricabile da Kaggle)
├── notebooks/
│   └── titanic_analysis.ipynb  # Workflow Jupyter completo e riproducibile
├── titanic_analysis.py     # Script Python standalone per la pipeline
├── requirements.txt        # Dipendenze Python
├── Dockerfile              # Ambiente runtime completamente replicabile
├── .dockerignore            # Esclude file inutili dal build context
├── .gitignore              # File da escludere da Git
└── README.md               # Questo file
```

---

## 🚀 Come eseguirlo

### Opzione 1: Docker (consigliato, zero setup locale)

1. **Clona il repository:**
   ```bash
   git clone https://github.com/dariolignana96/titanic-ml.git
   cd titanic-ml
   ```

2. **Scarica il dataset da Kaggle:**
   - Vai su [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
   - Scarica `train.csv` e posizionalo in `data/train.csv`

3. **Build e avvia con Docker:**
   ```bash
   docker build -t titanic-ml .
   docker run -p 8888:8888 titanic-ml
   ```
   
   Copia/incolla il link Jupyter (con token) dal terminale nel browser.

4. **Accedi a Jupyter Notebook:**
   - Troverai il workflow completo in `notebooks/titanic_analysis.ipynb`

---

### Opzione 2: Esecuzione manuale (Windows PowerShell / macOS / Linux)

1. **Clona il repository:**
   ```bash
   git clone https://github.com/dariolignana96/titanic-ml.git
   cd titanic-ml
   ```

2. **Scarica il dataset:**
   - Vai su [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic/data)
   - Scarica `train.csv` e posizionalo in `data/train.csv`

3. **Crea e attiva un virtual environment:**
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

4. **Installa le dipendenze:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Avvia Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```
   Poi apri il file `notebooks/titanic_analysis.ipynb` nel browser.

---

### Opzione 3: Esegui lo script standalone

Se vuoi solo eseguire l'analisi senza Jupyter:

```bash
python titanic_analysis.py
```

---

## 📦 Dipendenze principali

- `pandas` — manipolazione dati
- `numpy` — operazioni numeriche
- `matplotlib` — visualizzazione grafici
- `seaborn` — visualizzazione statistica
- `scikit-learn` — modelli ML e valutazione
- `jupyter` — notebook interattivi

_Tutte elencate nel file `requirements.txt`._

---

## 📊 Dataset

**Dataset Titanic** da [Kaggle: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)

### Licenza Dataset
- **Licenza:** CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0)
- **Source:** https://www.kaggle.com/competitions/titanic/data
- **Nota:** Il dataset **non è incluso** in questo repository per motivi di licenza
- **Utilizzo:** Scaricare direttamente da Kaggle per scopi educativi e di portfolio

---

## 🤖 Modello & Metriche

- **Modello:** Random Forest Classifier
- **Metriche:** Accuracy, Precision, Recall, F1-Score
- **Features utilizzate:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
- **Train/Test Split:** 80/20 con random_state=42

---

## 🤝 Note per recruiter / reviewer

Questo progetto è pensato per mostrare:

- ✅ **Data Exploration & EDA** — analisi approfondita del dataset
- ✅ **Data Cleaning & Preprocessing** — gestione valori mancanti, encoding
- ✅ **Feature Engineering** — selezione e trasformazione feature
- ✅ **Model Building & Evaluation** — training e validazione ML
- ✅ **Visualization** — grafici informativi con matplotlib/seaborn
- ✅ **Reproducibility** — Jupyter notebook + script Python + Docker
- ✅ **Best Practices** — virtual env, requirements.txt, modularità

---

## 📜 Licenza e Crediti

### Codice del Progetto
Il codice Python (`titanic_analysis.py`, notebook Jupyter), configurazione Docker e documentazione sono **originali** e disponibili per uso libero in contesti **open source** e **didattici**.

Puoi clonarlo, modificarlo e adattarlo per i tuoi esperimenti o per mostrarlo in colloquio.

### Dataset
- **Dataset:** [Kaggle Competition - Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/data)
- **Licenza Dataset:** CC BY-SA 4.0
- **Autore analisi:** [Dario Lignana](https://github.com/dariolignana96)

---

## 🔗 Link Utili

- [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic)
- [Documentazione Scikit-learn](https://scikit-learn.org/)
- [Documentazione Pandas](https://pandas.pydata.org/)
- [Documentazione Jupyter](https://jupyter.org/)
```