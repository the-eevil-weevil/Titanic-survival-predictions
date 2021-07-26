#!/usr/bin/env python
# coding: utf-8

# In[245]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[246]:


df = pd.read_csv(r'E:/ML Projects/titanic.csv')


# In[247]:


df.head()


# In[248]:


df.corrwith(df['Survived'])


# In[249]:


df.shape


# In[250]:


df.duplicated().sum()


# In[251]:


df.shape[0]


# In[252]:


df.isnull().sum()


# In[253]:


df['Age'] = df['Age'].fillna(df['Age'].mean())


# In[254]:


df = df.drop(['Cabin','SibSp'],axis=1)


# In[255]:


df.dtypes


# In[256]:


df = df.dropna(subset=['Embarked'])


# In[257]:


df.isnull().sum()


# In[258]:


df.isnull().sum()


# In[259]:


df.dtypes


# In[260]:


df = df.drop(['Ticket','Name'],axis=1)


# In[261]:


df1 = pd.get_dummies(df, columns=['Sex','Embarked',],drop_first=True)


# In[262]:


df1['Intercept'] = 1


# In[263]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as vif


# In[264]:


VIF = pd.DataFrame()


# In[265]:


VIF['Variables'] = df1.columns


# In[266]:


VIF['vif'] = [vif(df1.values,i) for i in range(df1.shape[1])]


# In[267]:


VIF


# In[268]:


df1 = df1.drop(['Intercept'],axis=1)


# In[269]:


x = df1.drop(['Survived'],axis=1).values


# In[270]:


y = df1['Survived']


# In[271]:


from sklearn.model_selection import train_test_split


# In[272]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25,
                                                random_state=0)


# In[273]:


from sklearn.linear_model import LogisticRegression


# In[274]:


regression = LogisticRegression()


# In[275]:


regression.fit(x_train,y_train)


# In[276]:


y_pred = regression.predict(x_test)


# In[277]:


from sklearn.metrics import r2_score, accuracy_score


# In[278]:


r2 = r2_score(y_test,y_pred)


# In[279]:


x.shape


# In[280]:


r2


# In[281]:


adj_r2 = (1-(1-r2)*(888/879))


# In[282]:


adj_r2


# In[283]:


from xgboost import XGBClassifier


# In[284]:


classifier = XGBClassifier()


# In[285]:


classifier.fit(x_train,y_train)


# In[286]:


y_pred1 = classifier.predict(x_test)


# In[287]:


accuracy_score(y_test,y_pred)


# In[288]:


from sklearn.tree import DecisionTreeClassifier


# In[289]:


classifier = DecisionTreeClassifier(criterion='entropy',)


# In[290]:


classifier.fit(x_train,y_train)


# In[291]:


y_pred3 = classifier.predict(x_test)


# In[292]:


accuracy_score(y_test,y_pred3)


# In[293]:


from sklearn.ensemble import RandomForestClassifier


# In[294]:


classifier = RandomForestClassifier(n_estimators=60, criterion='entropy',n_jobs=-1,max_depth=5)


# In[295]:


classifier.fit(x_train,y_train)


# In[296]:


y_pred4 = classifier.predict(x_test)


# In[297]:


accuracy_score(y_test,y_pred)


# In[298]:


from sklearn.model_selection import GridSearchCV


# In[186]:


parameters = {'n_estimators':[10,20,25,30,40,45,50,60,75,80,90,100],
             'criterion': ['gini','entropy'],
             'max_depth':[2,3,4,5]
             }


# In[201]:


grid_ = GridSearchCV(classifier, param_grid=parameters)


# In[202]:


grid = grid_.fit(x_train,y_train)


# In[203]:


grid.best_estimator_


# In[204]:


grid.best_score_


# In[299]:


classifier2 = RandomForestClassifier(criterion='entropy',max_depth=5, n_estimators=45,
                                    n_jobs=-1, )


# In[300]:


classifier2.fit(x_train,y_train)


# In[301]:


y_pred6 = classifier2.predict(x_test)


# In[302]:


accuracy_score(y_test,y_pred6)


# In[191]:


from sklearn.naive_bayes import GaussianNB


# In[127]:


classifier1 = GaussianNB()


# In[128]:


classifier1.fit(x_train,y_train)


# In[131]:


y_pred5 = classifier1.predict(x_test)


# In[132]:


accuracy_score(y_test,y_pred5)


# In[ ]:




