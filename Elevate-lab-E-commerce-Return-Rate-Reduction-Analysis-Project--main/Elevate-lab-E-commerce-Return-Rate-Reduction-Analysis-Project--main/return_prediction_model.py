
# E-commerce Return Rate Prediction Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("your_merged_orders.csv")

# Encode categorical variables
le = LabelEncoder()
for col in ["category", "supplier", "marketing_channel"]:
    df[col] = le.fit_transform(df[col])

X = df[["price","quantity","category","supplier","marketing_channel"]]
y = df["is_returned"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

df["return_probability"] = model.predict_proba(X)[:,1]

high_risk = df[df["return_probability"] > 0.6]
high_risk.to_csv("high_risk_products.csv",index=False)

print("Model Training Complete. High Risk CSV Generated.")
