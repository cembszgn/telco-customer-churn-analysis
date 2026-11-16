"""Telco Customer Churn — uçtan uca makine öğrenmesi çalışması. Sentetik veri üretmez."""
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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,RocCurveDisplay,ConfusionMatrixDisplay
ROOT=Path(__file__).resolve().parents[1]; RAW_DIR=ROOT/"data"/"raw"; OUT_DIR=ROOT/"outputs"; PLOT_DIR=OUT_DIR/"grafikler"; DATA_PATH=RAW_DIR/"Telco-Customer-Churn.csv"
DATA_URL="https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
def load_data():
 RAW_DIR.mkdir(parents=True,exist_ok=True)
 if not DATA_PATH.exists(): print("IBM Telco verisi indiriliyor..."); urllib.request.urlretrieve(DATA_URL,DATA_PATH)
 df=pd.read_csv(DATA_PATH)
 if "Churn" not in df.columns: raise ValueError("Churn hedef sütunu bulunamadı.")
 return df
def prepare_data(df):
 data=df.copy().drop(columns=["customerID"],errors="ignore")
 data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors="coerce")
 data=data.dropna(subset=["TotalCharges"]); data["Churn"]=data["Churn"].map({"No":0,"Yes":1}); data=data.dropna(subset=["Churn"])
 return data,data.drop(columns=["Churn"]),data["Churn"].astype(int)
def build_models(X):
 categorical=X.select_dtypes(include=["object","category"]).columns.tolist(); numeric=X.select_dtypes(include=["number","bool"]).columns.tolist()
 num=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
 cat=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
 prep=ColumnTransformer([("num",num,numeric),("cat",cat,categorical)])
 return {"Logistic Regression":Pipeline([("preprocess",prep),("model",LogisticRegression(max_iter=1500,class_weight="balanced",random_state=42))]),"Random Forest":Pipeline([("preprocess",prep),("model",RandomForestClassifier(n_estimators=350,max_depth=12,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1))])}
def main():
 OUT_DIR.mkdir(parents=True,exist_ok=True); PLOT_DIR.mkdir(parents=True,exist_ok=True)
 raw=load_data(); print("Ham veri boyutu:",raw.shape)
 data,X,y=prepare_data(raw); print("Temiz veri boyutu:",data.shape)
 X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
 models=build_models(X); metrics=[]; fitted={}; fig,ax=plt.subplots(figsize=(8,6))
 for name,model in models.items():
  model.fit(X_train,y_train); pred=model.predict(X_test); prob=model.predict_proba(X_test)[:,1]
  metrics.append({"Model":name,"Accuracy":accuracy_score(y_test,pred),"Precision":precision_score(y_test,pred,zero_division=0),"Recall":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),"ROC_AUC":roc_auc_score(y_test,prob)})
  fitted[name]=model; RocCurveDisplay.from_predictions(y_test,prob,name=name,ax=ax)
  cm=confusion_matrix(y_test,pred); cf,ca=plt.subplots(figsize=(5,4)); ConfusionMatrixDisplay(cm,display_labels=["Kalmış","Ayrılmış"]).plot(cmap="Greens",values_format="d",ax=ca,colorbar=False); ca.set_title(f"{name} — Confusion Matrix"); cf.tight_layout(); cf.savefig(PLOT_DIR/f"confusion_{name.lower().replace(' ','_')}.png",dpi=160); plt.close(cf)
  print(f"\n{name}\n",classification_report(y_test,pred,target_names=["Kalmış","Ayrılmış"],zero_division=0))
 ax.plot([0,1],[0,1],"--",color="gray",label="Rastgele tahmin"); ax.set_title("ROC Eğrileri"); ax.legend(); fig.tight_layout(); fig.savefig(PLOT_DIR/"roc_curves.png",dpi=160); plt.close(fig)
 results=pd.DataFrame(metrics).sort_values("F1",ascending=False); results.to_csv(OUT_DIR/"model_karsilastirma.csv",index=False); print(results.to_string(index=False))
 best_name=results.iloc[0]["Model"]; best=fitted[best_name]; joblib.dump(best,OUT_DIR/"en_iyi_model.joblib")
 pd.DataFrame({"Gerçek":y_test,"Tahmin":best.predict(X_test)}).to_csv(OUT_DIR/"test_tahminleri.csv",index=False)
 errors=X_test.copy(); errors["Gerçek"]=y_test; errors["Tahmin"]=best.predict(X_test); errors=errors[errors["Gerçek"]!=errors["Tahmin"]]; errors.to_csv(OUT_DIR/"yanlis_tahminler.csv",index=False)
 print("Seçilen model:",best_name,"| Test:",len(y_test),"| Hatalı:",len(errors))
if __name__=="__main__": main()
