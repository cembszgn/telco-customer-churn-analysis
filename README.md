# Telco Customer Churn — Makine Öğrenmesi

## Gereksinimler ve çalıştırma
Python 3.10+; `pip install -r requirements.txt`; `python src/train.py`. Veri CSV pakete dahildir. Notebook: `notebooks/Telco_Churn_Analizi.ipynb`.

## İçerik
- IBM Telco Customer Churn CSV
- Veri keşfi ve ön işleme
- Logistic Regression ve Random Forest karşılaştırması
- Accuracy, Precision, Recall, F1, ROC-AUC; confusion matrix, ROC eğrileri ve yanlış tahmin analizi
- PDF/DOCX rapor ve sunum

## Gerçek sonuç özeti
Random Forest: Accuracy 0.768, Precision 0.548, Recall 0.717, F1 0.621, ROC-AUC 0.831. Logistic Regression: 0.726, 0.490, 0.797, 0.607, 0.835. En yüksek F1'e göre Random Forest seçildi; churn recall açısından Logistic Regression daha yüksek.

Kaynak: IBM Telco Customer Churn açık örnek veri seti. Bu proje karar destek amaçlı eğitim çalışmasıdır.
