import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from pathlib import Path

var_samples = 200000
var_features = 20
var_informativefeatures = 15 #4/3 - Features/Informatives
myrst = 51

def Data():
    project_dir = Path(__file__).resolve().parent.parent
    dir_data = project_dir / "data"
    dir_data.mkdir(parents=True, exist_ok=True)
    print("[Data directory found]")
    return dir_data

def DataGeneration_Classification():
    file_name = 'dataset(classification).csv'

    X, y = make_classification(
        n_samples=var_samples,
        n_features=var_features, 
        n_informative=var_informativefeatures, 
        n_classes=2, 
        shuffle=False, 
        random_state=myrst
    )

    col_names = []
    for i in range(var_informativefeatures):
        col_names.append(f'Inf{i+1}')
    for i in range(var_features-var_informativefeatures):
        col_names.append(f'Noi{i+1}')

    df = pd.DataFrame(X, columns=col_names)
    df['Y'] = y

    df.to_csv(Data() / file_name, index=False)

    print("[Data for classification generated successfully]")

def DataGeneration_Regression():
    file_name = 'dataset(regression).csv'

    X, y = make_regression(
        n_samples=var_samples,
        n_features=var_features, 
        n_informative=var_informativefeatures, 
        shuffle=False, 
        random_state=myrst
    )

    col_names = []
    for i in range(var_informativefeatures):
        col_names.append(f'Inf{i+1}')
    for i in range(var_features-var_informativefeatures):
        col_names.append(f'Noi{i+1}')

    df = pd.DataFrame(X, columns=col_names)
    df['Y'] = y

    df.to_csv(Data() / file_name, index=False)

    print("[Data for regression generated successfully]")

if __name__ == "__main__":
    DataGeneration_Classification()
    DataGeneration_Regression()