# SMOTE Results: Before vs After Comparison

## 🎯 Objective
Handle class imbalance to improve detection of churned customers (minority class).

---

## 📊 Results Comparison

### **Without SMOTE (Original)**

| Metric | Value |
|--------|-------|
| **Training Samples** | 8,000 |
| **Class Distribution** | Imbalanced (79.6% stayed / 20.4% churned) |
| **Best Model** | Gradient Boosting |
| **Accuracy** | 86.75% |
| **F1 Score** | 0.594 |
| **Precision** | 78.86% |
| **Recall** | **47.67%** ⚠️ (Missing 52% of churned customers!) |
| **ROC-AUC** | 0.867 |

### **With SMOTE (Balanced)**

| Metric | Value |
|--------|-------|
| **Training Samples** | 12,740 (+59% synthetic samples) |
| **Class Distribution** | Balanced (50% stayed / 50% churned) |
| **Best Model** | Gradient Boosting |
| **Accuracy** | 80.60% |
| **F1 Score** | 0.589 |
| **Precision** | 51.77% |
| **Recall** | **68.30%** ✅ (Only missing 32% now!) |
| **ROC-AUC** | 0.853 |

---

## 🔥 Key Improvements

### ✅ **Massive Recall Improvement**
- **Before**: 47.67% recall → Missing **213 out of 407** churned customers
- **After**: 68.30% recall → Missing only **129 out of 407** churned customers
- **Impact**: **Catching 84 more churned customers!** (43% improvement in detection)

### ⚖️ **Trade-offs**
- **Accuracy**: Decreased from 86.75% → 80.60% (-6.15%)
  - This is acceptable because we're now better at catching the important minority class
- **Precision**: Decreased from 78.86% → 51.77%
  - More false positives, but better than missing real churned customers
- **F1 Score**: Nearly identical (0.594 → 0.589)
  - Better balance between precision and recall

---

## 💡 Why This Matters

### For Churn Prediction:
- **Missing a churned customer** is **more costly** than a false alarm
- If we predict someone will churn, we can take action (offer discount, call customer service, etc.)
- If we miss a churned customer, they're gone forever

### Business Impact:
With **68.30% recall**:
- Out of 407 actual churned customers
- We now correctly identify **278 customers** who are about to churn
- We can intervene and potentially save them
- Before SMOTE, we only caught 194 customers

**Net Result**: We can now reach out to **84 more at-risk customers** before they leave!

---

## 📈 Model Comparison (With SMOTE)

| Rank | Model | F1 Score | Recall | Precision | Accuracy |
|------|-------|----------|--------|-----------|----------|
| 🥇 1 | **Gradient Boosting** | **0.589** | **68.30%** | 51.77% | 80.60% |
| 🥈 2 | CatBoost | 0.583 | 64.86% | 53.01% | 81.15% |
| 🥉 3 | Random Forest | 0.581 | 64.13% | 53.16% | 81.20% |
| 4 | XGBoost | 0.576 | 62.90% | 53.11% | 81.15% |
| 5 | AdaBoost | 0.573 | 69.78% | 48.55% | 78.80% |
| 6 | KNeighbors | 0.534 | 64.13% | 45.71% | 77.20% |
| 7 | Decision Tree | 0.487 | 59.71% | 41.12% | 74.40% |
| 8 | Logistic Regression | 0.483 | 61.92% | 39.62% | 73.05% |

---

## 🏆 Best Model Performance (Gradient Boosting with SMOTE)

**Confusion Matrix**:
```
                Predicted
                Stayed  Churned
Actual Stayed    1334     259
       Churned    129     278
```

- **True Positives**: 278 (correctly identified churned customers)
- **False Negatives**: 129 (churned customers we missed)
- **False Positives**: 259 (false alarms)
- **True Negatives**: 1334 (correctly identified staying customers)

**Classification Report**:
```
              precision    recall  f1-score   support
  Stayed (0)       0.91      0.84      0.87      1593
 Churned (1)       0.52      0.68      0.59       407

    accuracy                           0.81      2000
```

---

## ✅ Recommendation

**Use SMOTE for this churn prediction problem** because:

1. ✅ **43% improvement in recall** - catches 84 more churned customers
2. ✅ **Balanced F1 score** - still 0.589 (almost identical to non-SMOTE 0.594)
3. ✅ **Better for business** - false positives are cheaper than missed churns
4. ✅ **Handles class imbalance** - goes from 80/20 split to 50/50 in training

The slight drop in accuracy is a worthwhile trade-off for **significantly better detection of at-risk customers**.

---

## 📁 Files Updated

- `src/feature_engineering/features.py` - Added `apply_smote()` method
- `artifacts/models/*.pkl` - All models retrained on SMOTE-balanced data
- `artifacts/metrics/model_comparison.csv` - Updated with SMOTE results
