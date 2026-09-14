import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

MODEL_PATH = "xgboost_model.pkl"


class XGBoostPredictor:

    def __init__(self):
        self.model = None

        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                print("🧠 Modelo XGBoost histórico cargado.")
            except Exception as e:
                print(f"⚠️ No se pudo cargar XGBoost: {e}")
                self.model = None

    def entrenar(self, df):

        print(f"📊 XGBoost recibió {len(df)} filas")

        if len(df) < 10:
            print("❌ Muy pocas filas para entrenar XGBoost")
            return False

        X = df.drop(columns=["objetivo"])
        y = df["objetivo"]

        # ============================================
        # 🔎 VERIFICAR LAS DOS CLASES
        # ============================================

        if len(y.unique()) < 2:
            print(
                "⚠️ No hay suficientes clases "
                "(0 y 1) para entrenar."
            )
            return False

        # ============================================
        # 📅 VALIDACIÓN TEMPORAL
        # ============================================

        punto_corte = int(len(X) * 0.80)

        X_train = X.iloc[:punto_corte]
        y_train = y.iloc[:punto_corte]

        X_test = X.iloc[punto_corte:]
        y_test = y.iloc[punto_corte:]

        if len(X_test) == 0:
            print("⚠️ No hay datos suficientes para validar.")
            return False

        # ============================================
        # 🌳 CREAR MODELO
        # ============================================

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )

        print("🚀 Entrenando XGBoost con histórico...")

        self.model.fit(
            X_train,
            y_train
        )

        print("✅ XGBoost terminó fit()")

        # ============================================
        # 🧪 VALIDACIÓN CON DATOS POSTERIORES
        # ============================================

        pred = self.model.predict(X_test)

        acc = accuracy_score(
            y_test,
            pred
        )

        # ============================================
        # 💾 GUARDAR MODELO
        # ============================================

        joblib.dump(
            self.model,
            MODEL_PATH
        )

        print(
            f"🧪 XGBoost | Precisión validación: "
            f"{acc:.2%}"
        )

        print(
            f"💾 Modelo XGBoost actualizado con "
            f"{len(X_train)} patrones históricos."
        )

        return acc

    def predecir(self, fila):

        if self.model is None:
            print(
                "⏳ Modelo XGBoost aún no entrenado."
            )
            return None

        try:

            prob = self.model.predict_proba(
                fila
            )[0][1]

            return float(prob)

        except Exception as e:

            print(
                f"⚠️ Error prediciendo con XGBoost: {e}"
            )

            return None
