
# Titanic - Machine Learning from Disaster

Analisi predittiva e classificazione della sopravvivenza dei passeggeri del Titanic tramite tecniche di Machine Learning.

## Descrizione

Questo progetto affronta la celebre sfida "Titanic: Machine Learning from Disaster" di Kaggle, con l'obiettivo di costruire un modello predittivo per stimare la probabilità di sopravvivenza di un passeggero sulla base di informazioni note a bordo.

Le fasi principali dell'analisi includono:
- **Esplorazione e pulizia del dataset**
- **Analisi dei pattern di sopravvivenza**
- **Preprocessing e feature engineering**
- **Costruzione e valutazione di un modello di Machine Learning**
- **Visualizzazione dei risultati**

## Struttura del repository

- `data/` — Contiene il dataset di training
- `notebooks/` — Notebook Jupyter con l'intero workflow di analisi
- `titanic_analysis.py` — Script Python standalone per la pipeline

## Principali step dell'analisi

1. **Setup e caricamento dati**  
   Importazione librerie, analisi esplorativa, descrizione e visualizzazione preliminare.
2. **Data Cleaning & Preprocessing**  
   Gestione valori mancanti, encoding delle variabili categoriche e selezione delle feature.
3. **Modellazione**  
   Addestramento di un classificatore Random Forest, suddivisione train/test e validazione.
4. **Valutazione e metriche**  
   Output di accuracy, classification report, visualizzazioni di confronto sulle predizioni.
5. **Conclusioni**  
   Evidenziazione dei risultati principali e spunti su come migliorare il modello.

## Come eseguire il progetto

1. Clona la repository:

   git clone https://github.com/dariolignana96/titanic-ml.git
   
2. Crea e attiva una virtual environment:

   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   
3. Installa le dipendenze:

   pip install -r requirements.txt
   
4. Avvia Jupyter Notebook:

   jupyter notebook

   e segui le celle nel file `notebooks/titanic_analysis.ipynb`

## Dipendenze principali

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

_Tutte le dipendenze sono elencate in `requirements.txt`._

## Dataset

Dataset Titanic disponibile su [Kaggle](https://www.kaggle.com/c/titanic).

## Credits

Progetto realizzato da [Dario Lignana](https://github.com/dariolignana96) — ispirato alla sfida Kaggle Titanic.

***

