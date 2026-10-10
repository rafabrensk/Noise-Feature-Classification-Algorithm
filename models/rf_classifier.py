#Random Forest Classifier model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

def DatasetPath():
    project_dir = Path(__file__).resolve().parent.parent
    dir_dataset = project_dir / "data" / "dataset(classification).csv"
    print("[Data directory found]")
    return dir_dataset

def train_rf_classifier():
    df = pd.read_csv(DatasetPath())
    print("[Dataset found]")

    X = df.drop('Y', axis=1)
    y = df['Y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=51)

    rf_classifier = RandomForestClassifier(n_estimators=128, random_state=51, oob_score=True)
    rf_classifier.fit(X_train, y_train)
    print("[Model trained successfully]")

    training_accuracy = accuracy_score(y_train, rf_classifier.predict(X_train))
    test_accuracy = accuracy_score(y_test, rf_classifier.predict(X_test))
    oob_score = rf_classifier.oob_score_
    
    df_importances = pd.DataFrame(
        {"Feature": X_train.columns, "Importance": rf_classifier.feature_importances_}
    ).sort_values(by="Importance", ascending=True)

    results = {
        "model": rf_classifier,
        "training_accuracy": training_accuracy,
        "test_accuracy": test_accuracy,
        "df_importances": df_importances,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "oob_score": oob_score,
    }
    print("[Metrics collected successfully]")

    return results

if __name__ == "__main__":
    train_rf_classifier()