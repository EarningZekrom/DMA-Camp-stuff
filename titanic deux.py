#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[72]:


train = pd.read_csv("train.csv")


# In[73]:


print (train)


# In[74]:


train.head(10)


# In[75]:


train.isnull()


# In[76]:


sns.heatmap(train.isnull())


# In[77]:


train.groupby("Pclass")["Age"].mean()


# In[78]:


sns.boxplot (x= "Pclass", y = "Age", data = train)


# In[79]:


def fill_age(passenger):
    age = passenger[0]
    pclass = passenger [1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age


# In[80]:


train["Age"] = train [["Age", "Pclass"]].apply(fill_age, axis = 1)


# In[81]:


train.drop("Cabin", axis = 1, inplace = True)


# In[82]:


sns.heatmap(train.isnull(), yticklabels = False, cbar = False, cmap = "viridis")


# In[83]:


gender = pd.get_dummies(train["Sex"], drop_first=True)


# In[84]:


gender


# In[85]:


train = pd.concat([train, gender], axis=1)


# In[86]:


embark = pd.get_dummies (train["Embarked"], drop_first = True)


# In[87]:


train = pd.concat([train, embark], axis=1)


# In[88]:


train.drop (["Sex", "Embarked", "Name", "Ticket"], axis = 1, inplace = True)


# In[89]:


train.head()


# In[90]:


sns.countplot(x= "Survived", hue = "male", data = train)


# In[91]:


sns.distplot(train["Age"].dropna(),bins=30,kde=False)


# In[92]:


sns.countplot(x="SibSp", data = train)


# In[93]:


sns.distplot(train["Fare"],bins=20,kde=False)


# In[94]:


from sklearn.model_selection import train_test_split


# In[95]:


X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived', axis=1), 
                                           train['Survived'], test_size = 0.30, 
                                           random_state=101)


# In[96]:


from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)


# In[97]:


from sklearn.metrics import classification_report
print (classification_report(y_test, predictions))


# In[ ]:





# In[ ]:





# In[ ]:




