#Basic insurance cost prediction using Linear Regression and Scikit-learn.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# reading data from file and storing it in dataframe
insurance_data = pd.read_csv("./Data/insurance.csv")


#Data Visualisation - create a scatter plot --- BMI vs Charges
sns.scatterplot(
    x=insurance_data["bmi"],
    y=insurance_data["charges"],
    hue=insurance_data["smoker"])
# plt.show()



#feature engineering
# smoker -> yes = 1, no = 0
# sex -> female = 1 , male = 0

X = insurance_data.drop(columns=["charges","region"])
y = insurance_data["charges"]
X["sex"] = X["sex"].map({"female":1, "male":0})
X["smoker"] = X["smoker"].map({"yes":1, "no":0})
print(X.head())

#Tain Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

#Train model
model = LinearRegression()
model.fit(X_train, y_train)



