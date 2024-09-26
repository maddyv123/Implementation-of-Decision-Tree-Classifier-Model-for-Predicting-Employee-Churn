# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.import pandas module and import the required data set.
2.Find the null values and count them.

3.Count number of left values.

4.From sklearn import LabelEncoder to convert string values to numerical values.

5.From sklearn.model_selection import train_test_split.

6.Assign the train dataset and test dataset.

7.From sklearn.tree import DecisionTreeClassifier.

8.Use criteria as entropy.

9.From sklearn import metrics.

10.Find the accuracy of our model and predict the require values.
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: MATHAVAN V
RegisterNumber:  212223110026
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
## DATASET:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/9d31af6c-2f10-4a3e-8021-2f49b721972d)
## df.head():
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/a05674af-c66d-4ed9-9de9-0615750b15a8)
## X.head():
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/0ce2b6a5-962c-4fe2-8c9b-cd0d78fc3726)
## df.info():
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/e3090a35-9cf7-4d44-a7ba-02ea557cb299)
## ISNULL:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/f8cdbcb7-fad8-4f14-9a3c-22cb444e2100)
## VALUE COUNTS:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/24dd0d8b-ce6a-4cbb-983d-de82c9817fce)
## DATA TRANSFORMED HEAD:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/f63a0da8-762b-4724-a060-3a0410052665)
## DECISION TREE:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/7c66e648-9450-4a4b-a4d3-79326adf200b)
## ACCURACY:
![image](https://github.com/Mythilidharman/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119104110/c2c72e9b-edac-4ae7-b0a8-945686ed2755)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
