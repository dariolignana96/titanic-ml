
# Titanic - Machine Learning from Disaster

Analisi predittiva e classificazione della sopravvivenza dei passeggeri del Titanic tramite tecniche di Machine Learning.

## Descrizione

Questo progetto affronta la sfida Kaggle "Titanic: Machine Learning from Disaster", con l'obiettivo di costruire un modello predittivo della probabilità di sopravvivenza dei passeggeri, utilizzando feature ingegnerizzate e un approccio data science completo.

**Le fasi principali:**
- Esplorazione e pulizia del dataset
- Analisi statistica e pattern di sopravvivenza
- Preprocessing, feature engineering e scaling
- Costruzione e validazione di modelli ML
- Visualizzazione e interpretazione dei risultati

## Struttura del repository

- `data/` — Dataset di training (non incluso per motivi di licenza, scaricabile da Kaggle)
- `notebooks/` — Workflow Jupyter completo e riproducibile
- `titanic_analysis.py` — Script Python standalone per la pipeline
- `Dockerfile` — Ambiente runtime completamente replicabile con Docker
- `.dockerignore` — Esclude virtualenv e file inutili dal build context

## Esecuzione con Docker (consigliato, zero setup locale!)

1. Clona la repository:

    git clone https://github.com/dariolignana96/titanic-ml.git
    cd titanic-ml

2. Build e avvia il progetto con Docker:

    docker build -t titanic-ml .
    docker run -p 8888:8888 titanic-ml
    
    Copia/incolla il link Jupyter (con token) dal terminale o dai log di Docker Desktop nel browser.

3. Accedi a Jupyter Notebook:
   - Troverai il workflow completo nella cartella `notebooks/`.

## Esecuzione manuale (senza Docker)

1. Crea e attiva un virtual environment:

    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    

2. Installa le dipendenze:

    pip install -r requirements.txt
    

3. Avvia Jupyter Notebook:

    jupyter notebook
    
    e segui le analisi nel file `notebooks/titanic_analysis.ipynb`

## Dipendenze principali

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter

_Tutte elencate nel file `requirements.txt`._

## Dataset

Dataset Titanic disponibile su [Kaggle](https://www.kaggle.com/c/titanic).

## Credits

Realizzato da [Dario Lignana](https://github.com/dariolignana96), ispirato dalla community Kaggle.

