#Task 1: Data Overview

#Importing libraries
"""

#for data exploration
import pandas as pd
import numpy as np
#for data visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#importing our dataset from local drive to colab
from google.colab import drive
drive.mount('/content/drive')

#naming our dataset as 'df'
df=pd.read_csv("/content/drive/MyDrive/Data Set.csv")

df

df.info()

df.shape

#finding Null values.
df.isnull().sum()

"""#Task 2: Gender Distribution"""

df.gender.value_counts()

plt.pie(df['gender'].value_counts(),labels = ['Male','Female'],autopct = "%0.2f",)
plt.show();

Gcount = df["gender"].value_counts()
Gcount.plot(kind = 'bar')
plt.show()

"""#Task 3: Descriptive Statistics"""

#df._get_numeric_data()
df.select_dtypes(include=['int64','float64'])

df.describe()

Columns = ['Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
print("Sum of each category is: ")
print(df[Columns].sum())

columns = ['Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
std_devs = df[columns].std()
print(std_devs)

plt.bar(columns, std_devs)
plt.xlabel('Columns')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Columns')
bar_colors = ['tab:red', 'tab:blue', 'tab:pink', 'tab:orange','tab:purple','tab:brown','tab:cyan']
plt.bar(columns, std_devs, color=bar_colors)
plt.show()

columns = ['Mutual_Funds', 'Equity_Market', 'Debentures', 'Government_Bonds', 'Fixed_Deposits', 'PPF', 'Gold']
mean = df[columns].mean()
print(mean)

plt.bar(columns, mean)
plt.xlabel('Columns')
plt.ylabel('Standard Deviation')
plt.title('Standard Deviation of Columns')
bar_colors = ['tab:red', 'tab:blue', 'tab:pink', 'tab:orange','tab:purple','tab:brown','tab:cyan']
plt.bar(columns, mean, color=bar_colors)
plt.show()

"""##Task 4: Most Preferred Investment Avenue"""

df.Investment_Avenues.value_counts()

df['Avenue'].value_counts()

"""The highest frequency or occurrence is of Mutual Funds

#Investment_Avenues distribution
"""

plt.pie(df['Avenue'].value_counts(),labels = ['Mutual Fund','Equity','Fixed Deposits','Public Provident Fund'],autopct = "%0.2f",)
plt.show();

"""##Task 5: Reasons for Investment
1. Reason_Equity
2. Reason_Mutual
3. Reason_Bonds
4. Reason_FD

#Reasons for Equity
"""

df.Reason_Equity.value_counts()

res1 = df.Reason_Equity.value_counts()
res1.plot(kind = 'bar')
plt.title('Reason_Equity Distribution')
bars = plt.bar(res1.index, res1, color=['tab:pink', 'tab:orange','tab:purple'])
plt.xlabel('Reason')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

df.Reason_Mutual.value_counts()

res1 = df.Reason_Mutual.value_counts()
res1.plot(kind = 'bar')
plt.title('Reason_Mutual Distribution')
bars = plt.bar(res1.index, res1, color=['tab:pink', 'tab:orange','tab:purple'])
plt.xlabel('Reason')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

df.Reason_Bonds.value_counts()

res1 = df.Reason_Bonds.value_counts()
res1.plot(kind = 'bar')
plt.title('Reason_Bonds Distribution')
bars = plt.bar(res1.index, res1, color=['tab:pink', 'tab:orange','tab:purple'])
plt.xlabel('Reason')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

df.Reason_FD.value_counts()

res1 = df.Reason_FD.value_counts()
res1.plot(kind = 'bar')
plt.title('Reason_FD Distribution')
bars = plt.bar(res1.index, res1, color=['tab:pink', 'tab:orange','tab:purple'])
plt.xlabel('Reason')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

"""##Task 6: Savings Objectives"""

df.rename(columns={'What are your savings objectives?': 'Savings_objectives'}, inplace=True)

df.Savings_objectives.value_counts()

res1 = df.Savings_objectives.value_counts()
res1.plot(kind = 'bar')
plt.title('Savings_objectives Distribution')
bars = plt.bar(res1.index, res1, color=['tab:pink', 'tab:orange','tab:purple'])
plt.xlabel('Reason')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

plt.pie(df['Savings_objectives'].value_counts(),labels = ['Retirement Plan','Health Care', 'Education'],autopct = "%0.2f",)
plt.show();
print("Maximum saving objective is 'Retirement Plan'")

"""##Task 7: Common Information Sources"""

Source = df.Source.value_counts()
print(Source)

Source.plot(kind = 'bar')
plt.title('Common Information Sources Distribution')
bars = plt.bar(Source.index, Source, color=['tab:pink', 'tab:orange','tab:purple', 'tab:cyan'])
plt.xlabel('Source')
plt.ylabel('Value')
for bar in bars:
    val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, val, int(val), ha='center', va='bottom')
plt.show()

plt.pie(df['Source'].value_counts(),labels = ['Financial Consultants','Newspapers and Magazines', 'Television', 'Internet'],autopct = "%0.2f",)
plt.show();

"""#Most common sources participants rely on."""

Source.nlargest(2)

"""#Most common sources participants rely on is Financial Consultants

##Task 8: Investment Duration
"""

df.Duration.value_counts()

plt.pie(df['Duration'].value_counts(),labels = ['3-5 years','1-3 years','Less than 1 year','More than 5 years'],autopct = "%0.2f",)
plt.show();

from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler

le = LabelEncoder()


df['Duration']=le.fit_transform(df['Duration'])

time = df.Duration.value_counts()
print(time)

# Average investment duration

data = {
    "Less than 1 year": 2,
    "1-3 years": 18,
    "3-5 years": 19,
    "More than 5 years": 1}

# Assumptions for average durations in each category(Mid-point)
assumptions = {
    "Less than 1 year": 0.5,
    "1-3 years": 2,
    "3-5 years": 4,
    "More than 5 years": 6}

total = sum(data.values())
weighted_sum = sum(data[duration] * assumptions[duration] for duration in data)
average_duration = weighted_sum / total

average_duration

"""##Task 9 - Expectations from Investments"""

# percentage growth investors expecting.
df.Expect.value_counts()

# Common expectations mentioned by participants.
df['Expect'].mode()

plt.pie(df['Expect'].value_counts(),labels = ['20%-30%','30%-40%','40%-50%'],autopct = "%0.2f",)
plt.show();
