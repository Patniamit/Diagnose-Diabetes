from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_preprocessed_data(dataset_path="./data/diab.csv"):
    df = pd.read_csv(dataset_path)
    # print(df.head())
    # print(len(df))

    # Replace Zeros
    zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in zero_not_accepted:
        df[column] = df[column].replace(0, np.NaN)
        mean = int(df[column].mean(skipna=True))
        df[column] = df[column].replace(np.NaN, mean)

    # split dataset
    X = df.iloc[:, 0:8]
    y = df.iloc[:, 8]
    # Map dataframe to encode values and put values into a numpy array
    encoded_labels = df['Outcome'].map(lambda x: 1 if x == 'Positive' else 0).values  # ham will be 0 and spam will be 1

    return X, encoded_labels


def train_and_store_model(dataset_path="./data/diab.csv", model_path="./pretrained_models/diabetes.pkl", max_k=30):

    best_model = None
    best_accuracy = 0.0

    X, encoded_labels = load_preprocessed_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, random_state=0, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    for k in range(max_k):
        model = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(k, accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    model_data = pickle.dumps(best_model)
    print(model_data)
    with open(model_path, "wb") as f:
        f.write(model_data)


def predict_diabetes(dataset_path="./data/diab.csv", model_path="./pretrained_models/diabetes.pkl"):

    with open(model_path, "rb") as f:
        model = pickle.loads(f.read())

    X, encoded_labels = load_preprocessed_data(dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(X, encoded_labels, random_state=0, test_size=0.2)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    Pregnancies = input('\nPlease enter Pregnancies : ')
    Pregnancies = int(Pregnancies)
    Glucose = input('\nPlease enter Glucose : ')
    Glucose = int(Glucose)
    BloodPressure = input('\nPlease enter BloodPressure : ')
    BloodPressure = int(BloodPressure)
    SkinThickness = input('\nPlease enter SkinThickness : ')
    SkinThickness = int(SkinThickness)
    Insulin = input('\nPlease enter Insulin : ')
    Insulin = int(Insulin)
    BMI = input('\nPlease enter BMI : ')
    BMI = float(BMI)
    DiabetesPedigreeFunction = input('\nPlease enter DiabetesPedigreeFunction :')
    DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)
    Age = input('\nPlease enter Age : ')
    Age = int(Age)
    X_final = scaler.transform(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    result = model.predict(X_final)

    if result == 1:
        print('\nPatient is diabetic.')
    else:
        print('\nPatient is not diabetic.')
