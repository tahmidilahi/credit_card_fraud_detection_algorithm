# Credit Card Fraud Detection using Machine Learning Algorithms

Data Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## **Dataset Description**
| Key Attributes | Description |
|--------------|-----------|
| DISTANCE_FROM_HOME | Distance from home where the transaction occurred |
| DISTANCE_FROM_LAST_TRANSACTION | Distance from the location of the last transaction |
| RATIO_TO_MEDIAN_PURCHASE_PRICE | The ratio of the transaction's purchase price to the median purchase price |
| REPEAT_RETAILER | Indicates whether the transaction occurred with the same retailer as a previous one |
| USED_CHIP | Indicates whether the transaction was made using a chip (credit card) |
| USED_PIN_NUMBER | Indicates whether the transaction involved using a PIN |
| ONLINE_ORDER | Indicates whether the transaction was made online |
| FRAUD | Indicates whether the transaction is fraudulent | 

This project investigates fraudulent activities within a massive dataset of one million credit card transactions. By leveraging machine learning algorithms, we aim to identify key features associated with fraudulent transactions and develop effective prediction models for fraud detection.

## **Data Preparation and Feature Engineering**
Here's an outline of the techniques employed:
  | Feature | Description |
  | ------ |------- |
  | Distance Measures | We calculated DISTANCE_RATIO to capture changes in transaction locations compared to the previous one. This helps identify unusual travel patterns potentially indicative of fraud | 
  | Purchase Price Normalization | To ensure consistency across transactions with varying amounts, RATIO_TO_MEDIAN_PURCHASE_PRICE was normalized using min-max scaling. This creates a standardized RATIO_TO_MEDIAN_PURCHASE_NORMALIZED feature |
  | Distance from Home Standardization | DISTANCE_FROM_HOME was standardized by subtracting the mean and dividing by the standard deviation. The resulting DISTANCE_NORMALIZED feature has a mean of 0 and a standard deviation of 1 |
  | Distance Categorization | To analyze distance-related trends more effectively, the standardized DISTANCE_NORMALIZED feature was segmented into quartiles, forming DISTANCE_CATEGORY. This allows for the creation of binary indicators (DISTANCE_NEAR,               DISTANCE_MEDIUM, DISTANCE_FAR, DISTANCE_VERY_FAR) representing different distance ranges |
  | Column Reordering | For improved model compatibility and readability, the FRAUD (target variable) column was moved to the last position |

## **Data Exploration**
A comprehensive data analysis was conducted on a 50,000-transaction subset chosen randomly. This exploration provided insights into the characteristics and distribution of various variables, particularly those related to fraudulent transactions (FRAUD = 1).

Fraudulent Transaction Analysis:
- Subset Overview: 50,000 transactions were randomly selected from the original dataset for this analysis 
  
Fraudulent Transaction Distribution:
- Total fraudulent transactions: **4,309** 
- Total non-fraudulent transactions: **45,691** 
- Fraudulent transaction rate in the subset: **9.43%**

**Bar Plot of Fraudulent vs. Non-Fraudulent Transactions by Purchased Method**
![1](https://github.com/tahmidilahi/credit_card_fraud_detection_algorithm/assets/170153132/95745d8e-a524-46c0-a69a-ef9f6ac7dc22)

**Pie Diagram of Fraudulent Transactions Percentage by Purchased Methods**
![2](https://github.com/tahmidilahi/credit_card_fraud_detection_algorithm/assets/170153132/3f03b51b-d3f9-4985-956c-60b05a810594)

**Histogram of Distance from Home (Values < 50) vs Transaction Types**
![3](https://github.com/tahmidilahi/credit_card_fraud_detection_algorithm/assets/170153132/f2cb969a-4609-48ad-9100-0954899ccf7e)

**Correlation Heatmap**
![4](https://github.com/tahmidilahi/credit_card_fraud_detection_algorithm/assets/170153132/e6b6b55b-a62d-4ab3-a6d4-061af44fbc2d)

## **Accuracy Comparison of Different Machine Learning Algorithms**
![5](https://github.com/tahmidilahi/credit_card_fraud_detection_algorithm/assets/170153132/ceaa4f12-69ab-449d-85a7-ecb99d3552cd)

The list of predictors used to determine the target variable (FRAUD) includes:

- DISTANCE_FROM_HOME: Captures the cardholder's distance from their registered address during a transaction.
- DISTANCE_FROM_LAST_TRANSACTION: Analyzes changes in location between consecutive transactions.
- ONLINE_ORDER: Indicates whether the transaction occurred online.
- RATIO_TO_MEDIAN_PURCHASE_NORMALIZED: Represents the ratio of a transaction amount to the median purchase amount, normalized for consistency.
- DISTANCE_VERY_FAR: Categorical feature indicating transactions very far from the cardholder's home.
  
Our analysis revealed that the Decision Tree classifier achieved the highest overall accuracy (96.75%) for fraud detection. However, it's crucial to consider class imbalance. While all models excelled at identifying non-fraudulent transactions, they exhibited varying performance for fraudulent cases. The Decision Tree again displayed a commendable balance, with a precision of 0.82 and recall of 0.81 for fraudulent transactions. Logistic Regression also demonstrated strength in this area (precision: 0.79).

These findings highlight the trade-offs between different algorithms and the importance of considering class imbalance in fraud detection.
The project effectively demonstrates the potential of machine learning for combating financial fraud, with the Decision Tree model emerging as a top performer in this specific dataset.
