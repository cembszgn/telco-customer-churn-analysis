from pathlib import Path
import urllib.request

url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
target = Path(__file__).resolve().parents[1] / "data" / "raw" / "Telco-Customer-Churn.csv"
target.parent.mkdir(parents=True, exist_ok=True)
if not target.exists():
    urllib.request.urlretrieve(url, target)
print(f"Veri seti hazır: {target}")
