# E-Commerce Delay Intelligence Pipeline

A production-ready machine learning and analytics pipeline designed to predict shipment delivery times, classify delay risks, segment regional fulfillment hubs, and generate actionable operational alerts for e-commerce logistics management.

---

## 1. How to Run

Follow these exact steps in your Windows Command Prompt or PowerShell (`C:\Users\USER\ecom-delivery-intelligence`) using Python 3.10:

### Step 1: Install Dependencies
```cmd
python -m pip install -r requirements.txt

###Step2: Execute the Pipeline Sequence
Generate Synthetic Data:
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe generate_data.py

Run Data Foundation & EDA:
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe part_a_clean_eda.py

Run Delivery-Hours Regression (Part B):
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe part_b_regression.py

Run Delay Classifier & Threshold Optimization (Part C):
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe part_c_classification.py

Run Hub Segmentation & Tier Audit (Part D):
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe part_d_segments.py

Serialize Model Package:
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe package_model.py

Execute Batch Inference:
C:\Users\USER\AppData\Local\Programs\Python\Python310\python.exe predict_delay.py