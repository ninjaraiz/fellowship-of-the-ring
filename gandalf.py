import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import random

from .EarendilsLight import EarendilsLight

class GANDALF(XGBClassifier):
    
    """
    🧙‍♂️ Gandalf the Predictor
    Clasificador XGBoost para predecir el cluster GMM a partir de (x, z, aoa, mach).
    """

    light = EarendilsLight(__name__)

    @classmethod
    def some_light(cls, name=None):
        """Atajo a Eärendil's Light."""
        return cls.light.help(name)

    def __init__(self, **kwargs):
        super().__init__(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            subsample=kwargs.get("subsample", 0.8),
            colsample_bytree=kwargs.get("colsample_bytree", 0.8),
            random_state=kwargs.get("random_state", 42),
            use_label_encoder=False,
            eval_metric="mlogloss",
            **{k: v for k, v in kwargs.items() if k not in [
                "n_estimators", "max_depth", "learning_rate",
                "subsample", "colsample_bytree", "random_state"
            ]}
        )
        
        self.features = None
        self.target = None
        self.is_trained = False

    # --- Entrenamiento ---
    def train(self, df_data: pd.DataFrame, features=None, target="clusters_GMM"):
        if features is None:
            features = ["x", "z", "aoa", "mach"]

        if target not in df_data.columns:
            raise ValueError(f"Target '{target}' not found in DataFrame.")

        X = df_data[features].values
        y = df_data[target].values

        print(f"🧙‍♂️ Entrenando Gandalf con {len(X)} muestras y {len(features)} features...")
        self.fit(X, y)
        self.features = features
        self.target = target
        self.is_trained = True
        print("✅ Gandalf ha aprendido a discernir los clusters GMM.")

    # --- Predicción ---
    def predict_clusters(self, df_new: pd.DataFrame):
        if not self.is_trained:
            raise RuntimeError("Gandalf aún no ha sido entrenado. Usa .train() primero.")

        missing_feats = [f for f in self.features if f not in df_new.columns]
        if missing_feats:
            raise ValueError(f"Faltan columnas necesarias para predecir: {missing_feats}")

        X_new = df_new[self.features].values
        preds = self.predict(X_new)
        df_pred = df_new.copy()
        df_pred["cluster_pred"] = preds
        return df_pred

    # --- Evaluación ---
    def evaluate(self, df_test: pd.DataFrame):
        """
        Evalúa el rendimiento del modelo sobre un conjunto de test.
        Muestra métricas de clasificación y la matriz de confusión.
        """
        if not self.is_trained:
            raise RuntimeError("Gandalf no ha sido entrenado.")

        df_pred = self.predict_clusters(df_test)
        y_true = df_test[self.target].values
        y_pred = df_pred["cluster_pred"].values

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")

        print("\n📊 Resultados del test:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\n" + classification_report(y_true, y_pred))

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Matriz de confusión — Gandalf")
        plt.xlabel("Predicho")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.show()

        return {"accuracy": acc, "f1": f1, "confusion_matrix": cm}

    # --- División por condiciones (aoa, mach) ---
    @staticmethod
    def split_by_conditions(df_data, group_cols=("aoa", "mach"), test_frac=0.25, random_state=42):
        """
        Divide el DataFrame en entrenamiento y test por combinaciones únicas de (aoa, mach).
        """
        unique_groups = df_data[list(group_cols)].drop_duplicates().values.tolist()
        random.seed(random_state)
        random.shuffle(unique_groups)

        n_test = max(1, int(len(unique_groups) * test_frac))
        test_groups = unique_groups[:n_test]
        train_groups = unique_groups[n_test:]

        cond_train = df_data[list(group_cols)].apply(tuple, axis=1).isin([tuple(g) for g in train_groups])
        cond_test = df_data[list(group_cols)].apply(tuple, axis=1).isin([tuple(g) for g in test_groups])

        df_train = df_data[cond_train].copy()
        df_test = df_data[cond_test].copy()

        print(f"✂️ División por condiciones: {len(train_groups)} grupos de entrenamiento, {len(test_groups)} de test.")
        return df_train, df_test

    # --- Importancia de características ---
    def plot_feature_importance(self):
        """
        Muestra la importancia relativa de cada feature del modelo Gandalf.
        """
        if not self.is_trained:
            raise RuntimeError("Gandalf aún no ha sido entrenado.")

        booster = self.get_booster()
        importance = booster.get_score(importance_type="weight")
        if not importance:
            print("⚠️ No hay información de importancia disponible.")
            return

        # Ordenar por importancia
        sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        features = list(sorted_importance.keys())
        values = list(sorted_importance.values())

        plt.figure(figsize=(7, 4))
        plt.bar(features, values, color="darkgreen", alpha=0.7)
        plt.title("🧙‍♂️ Importancia de características según Gandalf")
        plt.ylabel("Importancia relativa")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()