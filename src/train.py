"""
Telco Customer Churn — uçtan uca makine öğrenmesi çalışması.
Gerçek IBM örnek veri setini ilk çalıştırmada indirir; sentetik veri üretmez.
"""
from pathlib import Path
import urllib.request
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "outputs"
PLOT_DIR = OUT_DIR / "grafikler"
DATA_PATH = RAW_DIR / "Telco-Customer-Churn.csv"
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def load_data():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        print("IBM Telco Customer Churn veri seti indiriliyor...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    if "Churn" not in df.columns:
        raise ValueError("Beklenen Churn hedef sütunu bulunamadı. Veri kaynağını kontrol edin.")
    return df

def prepare_data(df):
    data = df.copy()
    # customerID yalnızca tanımlayıcıdır; tahmin özelliği olarak kullanılmaz.
    data = data.drop(columns=["customerID"], errors="ignore")
    # TotalCharges sayısal görünse de boşluklar nedeniyle metin gelebilir.
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    # IBM örnek verisinde tenure=0 olan 11 kaydın TotalCharges alanı boştur.
    data = data.dropna(subset=["TotalCharges"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=["Churn"])
    X = data.drop(columns=["Churn"])
    y = data["Churn"].astype(int)
    return data, X, y

def build_models(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])
    models = {
        "Logistic Regression": Pipeline([
            ("preprocess", prep),
            ("model", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": Pipeline([
            ("preprocess", prep),
            ("model", RandomForestClassifier(
                n_estimators=350, max_depth=12, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            ))
        ])
    }
    return models

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_data()
    print("Ham veri boyutu:", raw.shape)
    print("Ham veri eksik değerleri:\n", raw.isna().sum().sort_values(ascending=False).head(10))
    data, X, y = prepare_data(raw)
    print("Temiz veri boyutu:", data.shape)
    print("Hedef dağılımı:\n", y.value_counts().rename(index={0:"Kalmış",1:"Ayrılmış"}))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = build_models(X)
    metrics, fitted = [], {}
    roc_fig, roc_ax = plt.subplots(figsize=(8,6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:,1]
        row = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob)
        }
        metrics.append(row)
        fitted[name] = model
        RocCurveDisplay.from_predictions(y_test, prob, name=name, ax=roc_ax)
        cm = confusion_matrix(y_test, pred)
        cm_fig, cm_ax = plt.subplots(figsize=(5,4))
        disp = ConfusionMatrixDisplay(cm, display_labels=["Kalmış","Ayrılmış"])
        disp.plot(cmap="Greens", values_format="d", ax=cm_ax, colorbar=False)
        cm_ax.set_title(f"{name} — Confusion Matrix")
        cm_fig.tight_layout()
        cm_fig.savefig(PLOT_DIR / f"confusion_{name.lower().replace(' ','_')}.png", dpi=160)
        plt.close(cm_fig)
        print(f"\n{name}\n", classification_report(y_test, pred, target_names=["Kalmış","Ayrılmış"], zero_division=0))
    roc_ax.plot([0,1],[0,1],"--",color="gray",label="Rastgele tahmin")
    roc_ax.set_title("ROC Eğrileri")
    roc_ax.legend()
    roc_fig.tight_layout()
    roc_fig.savefig(PLOT_DIR / "roc_curves.png", dpi=160)
    plt.close(roc_fig)

    results = pd.DataFrame(metrics).sort_values("F1", ascending=False)
    results.to_csv(OUT_DIR / "model_karsilastirma.csv", index=False)
    print("\nMODEL KARŞILAŞTIRMASI\n", results.to_string(index=False))

    # En yüksek F1 değerine sahip modeli seç.
    best_name = results.iloc[0]["Model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, OUT_DIR / "en_iyi_model.joblib")
    pd.DataFrame({"Gerçek": y_test, "Tahmin": best_model.predict(X_test)}).to_csv(
        OUT_DIR / "test_tahminleri.csv", index=False
    )

    # Hata analizi: yanlış sınıflandırılan kayıtlar.
    best_pred = best_model.predict(X_test)
    errors = X_test.copy()
    errors["Gerçek"] = y_test
    errors["Tahmin"] = best_pred
    errors = errors[errors["Gerçek"] != errors["Tahmin"]]
    errors.to_csv(OUT_DIR / "yanlis_tahminler.csv", index=False)
    print(f"\nSeçilen model: {best_name}")
    print(f"Test örneği: {len(y_test)} | Yanlış tahmin: {len(errors)}")
    print("Çıktılar:", OUT_DIR)

if __name__ == "__main__":
    main()
