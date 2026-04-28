# Titanic — Machine Learning from Disaster

Predictive classification of passenger survival using the Kaggle Titanic dataset.

## Overview

End-to-end data science pipeline covering exploratory analysis, preprocessing,
feature engineering, and model evaluation. Includes both a Jupyter notebook for
interactive exploration and a standalone Python script for reproducibility.

## Stack

- Python 3 — pandas, numpy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook
- Docker

## Setup

### Docker (recommended)

```bash
git clone https://github.com/dariolignana96/titanic-ml.git
cd titanic-ml
```

Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/titanic/data)
and place it in `data/train.csv`, then:

```bash
docker build -t titanic-ml .
docker run -p 8888:8888 titanic-ml
```

### Local

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```

Open `notebooks/titanic_analysis.ipynb`.

### Script only

```bash
python titanic_analysis.py
```

## Model

Random Forest Classifier — evaluated on accuracy, precision, recall, F1.
Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.

## License

Project code: MIT — see [LICENSE](LICENSE) for details.
Dataset: CC BY-SA 4.0 — download from [Kaggle](https://www.kaggle.com/competitions/titanic/data). Not included in this repository.