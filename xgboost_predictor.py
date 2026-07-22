import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = "xgboost_model.pkl"


class XGBoostPredictor:

    def __init__(self):
        self.model = None

        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)

    def entrenar(self, df):

        if len(df) < 100:
            return False

        X = df.drop(columns=["objetivo"])
        y = df["objetivo"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        pred = self.model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        joblib.dump(self.model, MODEL_PATH)

        print(f"✅ XGBoost entrenado | Precisión: {acc:.2%}")

        return acc

    def predecir(self, fila):

        if self.model is None:
            return None

        prob = self.model.predict_proba(fila)[0][1]

        return float(prob)
