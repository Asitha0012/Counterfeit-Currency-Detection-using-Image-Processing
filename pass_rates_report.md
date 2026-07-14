# Comprehensive Augmented Dataset Pass Rates Report

> **Note**: For Genuine notes, a 'Pass' means the system identified it as TRUE. For Counterfeit notes, a 'Pass' means the system successfully rejected it as FALSE.

## Genuine Notes Analysis

**Total Notes Evaluated**: 1500
**Overall Genuine Pass Rate**: 1429/1500 (95.27%)

### Pass Rate by Denomination

| Denomination | Passed | Total | Pass Percentage |
| :--- | :--- | :--- | :--- |
| LKR_500 | 475 | 500 | 95.00% |
| LKR_1000 | 474 | 500 | 94.80% |
| LKR_5000 | 480 | 500 | 96.00% |

### Pass Rate by Augmentation Category

| Category | Passed | Total | Pass Percentage |
| :--- | :--- | :--- | :--- |
| Brightness | 242 | 254 | 95.28% |
| Contrast | 243 | 251 | 96.81% |
| HSV | 312 | 317 | 98.42% |
| Luminance | 202 | 228 | 88.60% |
| Preset | 212 | 227 | 93.39% |
| Zigma | 218 | 223 | 97.76% |

---

## Counterfeit Notes Analysis

**Total Notes Evaluated**: 1500
**Overall Counterfeit Pass Rate**: 1472/1500 (98.13%)

### Pass Rate by Denomination

| Denomination | Passed | Total | Pass Percentage |
| :--- | :--- | :--- | :--- |
| LKR_500 | 500 | 500 | 100.00% |
| LKR_1000 | 489 | 500 | 97.80% |
| LKR_5000 | 483 | 500 | 96.60% |

### Pass Rate by Augmentation Category

| Category | Passed | Total | Pass Percentage |
| :--- | :--- | :--- | :--- |
| Brightness | 243 | 244 | 99.59% |
| Contrast | 243 | 245 | 99.18% |
| HSV | 243 | 249 | 97.59% |
| Luminance | 262 | 269 | 97.40% |
| Preset | 219 | 230 | 95.22% |
| Zigma | 262 | 263 | 99.62% |

---

## 3. Evaluation Metrics

To provide a standard scientific evaluation of the pipeline, we define the detection of a **Counterfeit** note as the **Positive Class (1)**, as this is the primary anomaly we are attempting to detect. The acceptance of a **Genuine** note is defined as the **Negative Class (0)**.

*   **True Positives (TP):** Counterfeit notes successfully rejected (1472)
*   **True Negatives (TN):** Genuine notes successfully passed (1429)
*   **False Positives (FP):** Genuine notes incorrectly rejected as fake (71)
*   **False Negatives (FN):** Counterfeit notes that incorrectly passed as genuine (28)

### Confusion Matrix

| | Predicted Genuine (0) | Predicted Counterfeit (1) |
| :--- | :--- | :--- |
| **Actual Genuine (0)** | TN: 1429 | FP: 71 |
| **Actual Counterfeit (1)** | FN: 28 | TP: 1472 |

### Standard Classification Metrics

*   **Accuracy:** (TP + TN) / Total = (1472 + 1429) / 3000 = **96.70%**
*   **Precision:** TP / (TP + FP) = 1472 / (1472 + 71) = **95.40%**
*   **Recall (Sensitivity):** TP / (TP + FN) = 1472 / (1472 + 28) = **98.13%**
*   **Specificity:** TN / (TN + FP) = 1429 / (1429 + 71) = **95.27%**
*   **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall) = 2 × (0.9540 × 0.9813) / (0.9540 + 0.9813) = **96.75%**

**Conclusion:** The system demonstrates extremely high Recall (98.13%), ensuring that almost all counterfeits are caught, while maintaining a strong Precision (95.40%) to minimize disruptions caused by rejecting worn-out genuine money. The excellent F1-Score (96.75%) validates the robust balance achieved by the Veto Gate and Flat Voting mechanisms.
