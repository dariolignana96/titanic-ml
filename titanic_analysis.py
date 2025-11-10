import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

pd.set_option('display.max_columns', None)
sns.set(style="whitegrid")

def load_data():
    return pd.read_csv('data/train.csv')

def analisi_missing(data):
    print("\nValori mancanti per colonna:")
    print(data.isnull().sum())
    sns.heatmap(data.isnull(), cbar=False, cmap="viridis")
    plt.title("Valori mancanti nel dataset")
    plt.show()

def plot_survived(data):
    print(data['Survived'].value_counts(normalize=True))
    sns.countplot(x='Survived', data=data)
    plt.title("Distribuzione Sopravvivenza")
    plt.show()
def plot_sex_survival(data):
    sns.countplot(x='Sex', hue='Survived', data=data)
    plt.title('Sopravvivenza in base al sesso')
    plt.show()

def plot_class_survival(data):
    sns.countplot(x='Pclass', hue='Survived', data=data)
    plt.title('Sopravvivenza in base alla classe')
    plt.show()

def plot_age_distribution(data):
    sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30)
    plt.title("Distribuzione età e sopravvivenza")
    plt.show()
    
def pulizia_dati(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data.drop('Cabin', axis=1, inplace=True)
    return data

def encoding(data):
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
    return data    
def prepare_data(data):
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']
    X = data[features]
    y = data['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def main():
    data = load_data()
    print(data.head())
    data.info()
    print(data.describe())

    analisi_missing(data)
    plot_survived(data)
    plot_sex_survival(data)
    plot_class_survival(data)
    plot_age_distribution(data)
    data = pulizia_dati(data)
    data = encoding(data)
    X_train, X_test, y_train, y_test = prepare_data(data)
    train_and_evaluate(X_train, X_test, y_train, y_test)
    
    
if __name__ == "__main__":
    main()
