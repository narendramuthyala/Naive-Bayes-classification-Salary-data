# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 23:31:29 2022

@author: 91995
"""

import pandas as pd
df = pd.read_csv("SalaryData_Train.csv")
df
df.shape
list(df)
df.dtypes
df.head(15)
df["educationno"].value_counts()
df["capitalgain"].value_counts()
df["capitalloss"].value_counts()
df["hoursperweek"].value_counts()


##########################################################################

df_1 = pd.read_csv("SalaryData_Test.csv")

df_1
df_1.dtypes

############################################################################3

df.isnull()
df.isnull().sum()


# finding duplicate rows
df.duplicated()
df[df.duplicated()].head()
df.duplicated().sum()

df = df.drop_duplicates()


df.duplicated()
df.duplicated().sum()
df[df.duplicated()]
df.columns.duplicated()

df.isna()
df.isna().sum()


###################################################################################

df.dtypes
import matplotlib.pyplot as plt

def  plot_boxplot(df,ft):
    df.boxplot(column=[ft])
    plt.grid(False)
    plt.show()
    
plot_boxplot(df,"age")

plot_boxplot(df,"educationno")
plot_boxplot(df,"capitalgain")
plot_boxplot(df,"capitalloss")
plot_boxplot(df,"hoursperweek")



def outliers(df,ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3-Q1
    
    lower_bound = Q1-3.5*IQR
    upper_bound = Q3+3.5*IQR
    ls = df.index[(df[ft]<lower_bound) | (df[ft] > upper_bound)]
    return ls


index_list = []
for feature in ["age","educationno","capitalgain","capitalloss","hoursperweek"]:
    index_list.extend(outliers(df,feature))

index_list
index_list


def remove(df,ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df

df_cleaned = remove(df,index_list)

df_cleaned
df_cleaned.shape


##########################################################################################3


df_cleaned["age"].hist()
df_cleaned["educationno"].hist()
df_cleaned["capitalgain"].hist()
df_cleaned["capitalloss"].hist()

df_cleaned["hoursperweek"].hist()



###########################################################################3

import matplotlib.pyplot as plt

df_cleaned.plot.scatter(x = ["age"],y = ["educationno"] ,color = "black")
df_cleaned.plot.scatter(x = ["capitalgain"],y = ["educationno"] ,color = "black")
df_cleaned.plot.scatter(x = ["age"],y = ["capitalgain"] ,color = "black")
df_cleaned.plot.scatter(x = ["hoursperweek"],y = ["educationno"] ,color = "black")


#######################################################################################


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_cleaned["workclass"] = LE.fit_transform(df_cleaned["workclass"])
df_cleaned["education"] = LE.fit_transform(df_cleaned["education"])
df_cleaned["maritalstatus"] = LE.fit_transform(df_cleaned["maritalstatus"])
df_cleaned["occupation"] = LE.fit_transform(df_cleaned["occupation"])
df_cleaned["relationship"] = LE.fit_transform(df_cleaned["relationship"])
df_cleaned["race"] = LE.fit_transform(df_cleaned["race"])
df_cleaned["sex"] = LE.fit_transform(df_cleaned["sex"])
df_cleaned["native"] = LE.fit_transform(df_cleaned["native"])
df_cleaned["Salary"] = LE.fit_transform(df_cleaned["Salary"])

df_cleaned.dtypes

#########################################################################################

#X = df_cleaned.drop("Salary",axis = 1)

#Y = df_cleaned["Salary"]

###############################################################

X = df_cleaned[["age","workclass","education","educationno","maritalstatus","occupation","relationship","race","sex","hoursperweek","native"]]

Y = df_cleaned["Salary"]






#############################################################################33
from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X,Y)
Y_pred = NB.predict(X)


from sklearn import metrics
cm = metrics.confusion_matrix(Y,Y_pred)

print(cm)


from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Y,Y_pred.round(3)))


################################################################33

from sklearn.metrics import log_loss
log_loss(Y,Y_pred).round(3)

###############################################################################333


from sklearn.linear_model import Ridge
Rg = Ridge(alpha = 500)
Rg.fit(X,Y)
y_pred = Rg.predict(X)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y,Y_pred)

print("Accuracy",accuracy.round(3))


Rg.coef_

pd.DataFrame(Rg.coef_)
pd.DataFrame(X.columns)
d1 = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(Rg.coef_)],axis = 1)
d1



######################################################################################################
################################################################################################


df_1 = pd.read_csv("SalaryData_Test.csv")

df_1
df_1.dtypes
df_1.shape


df_1.isnull()
df_1.isnull().sum()


# finding duplicate rows
df_1.duplicated()
df_1[df_1.duplicated()].head()
df_1.duplicated().sum()

df_1 = df_1.drop_duplicates()
df_1.shape

df_1.duplicated()
df_1.duplicated().sum()
df_1[df_1.duplicated()]
df_1.columns.duplicated()

######################################################################################################3



df_1.dtypes
import matplotlib.pyplot as plt

def  plot_boxplot(df_1,ft):
    df_1.boxplot(column=[ft])
    plt.grid(False)
    plt.show()
    
plot_boxplot(df_1,"age")

plot_boxplot(df_1,"educationno")
plot_boxplot(df_1,"capitalgain")
plot_boxplot(df_1,"capitalloss")
plot_boxplot(df_1,"hoursperweek")



def outliers(df_1,ft):
    Q1 = df_1[ft].quantile(0.25)
    Q3 = df_1[ft].quantile(0.75)
    IQR = Q3-Q1
    
    lower_bound = Q1-3.5*IQR
    upper_bound = Q3+3.5*IQR
    ls = df_1.index[(df_1[ft]<lower_bound) | (df_1[ft] > upper_bound)]
    return ls


index_list = []
for feature in ["age","educationno","capitalgain","capitalloss","hoursperweek"]:
    index_list.extend(outliers(df_1,feature))

index_list
index_list


def remove(df_1,ls):
    ls = sorted(set(ls))
    df_1 = df_1.drop(ls)
    return df_1

df_1_cleaned = remove(df_1,index_list)

df_1_cleaned
df_1_cleaned.shape

###########################################################################################


df_1_cleaned["age"].hist()
df_1_cleaned["educationno"].hist()
df_1_cleaned["capitalgain"].hist()
df_1_cleaned["capitalloss"].hist()

df_1_cleaned["hoursperweek"].hist()


############################################################################################


import matplotlib.pyplot as plt

df_1_cleaned.plot.scatter(x = ["age"],y = ["educationno"] ,color = "black")
df_1_cleaned.plot.scatter(x = ["capitalgain"],y = ["educationno"] ,color = "black")
df_1_cleaned.plot.scatter(x = ["age"],y = ["capitalgain"] ,color = "black")
df_1_cleaned.plot.scatter(x = ["hoursperweek"],y = ["educationno"] ,color = "black")

###########################################################################################


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df_1_cleaned["workclass"] = LE.fit_transform(df_1_cleaned["workclass"])
df_1_cleaned["education"] = LE.fit_transform(df_1_cleaned["education"])
df_1_cleaned["maritalstatus"] = LE.fit_transform(df_1_cleaned["maritalstatus"])
df_1_cleaned["occupation"] = LE.fit_transform(df_1_cleaned["occupation"])
df_1_cleaned["relationship"] = LE.fit_transform(df_1_cleaned["relationship"])
df_1_cleaned["race"] = LE.fit_transform(df_1_cleaned["race"])
df_1_cleaned["sex"] = LE.fit_transform(df_1_cleaned["sex"])
df_1_cleaned["native"] = LE.fit_transform(df_1_cleaned["native"])
df_1_cleaned["Salary"] = LE.fit_transform(df_1_cleaned["Salary"])

df_1_cleaned.dtypes


############################################################################################


X1 = df_1_cleaned.drop("Salary",axis = 1)

Y1 = df_1_cleaned["Salary"]

###############################################################

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X1,Y1)
Y_pred1 = NB.predict(X1)

from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Y1,Y_pred1.round(3)))


################################################################33

from sklearn.metrics import log_loss
log_loss(Y1,Y_pred1).round(3)




































































