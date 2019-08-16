#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[93]:


train = pd.read_csv("StudentsPerformance.csv")


# In[94]:


ids=[]
for i in range(0,len(train["lunch"])):
    ids.append(i)
print(ids)
train["ID"]=ids


# In[95]:


print (train)


# In[96]:


train.head (10)


# In[97]:


train.drop ("gender", axis = 1, inplace = True)


# In[98]:


train.head()


# In[99]:


train.drop ("race/ethnicity", axis = 1, inplace = True)


# In[100]:


train.drop ("parental level of education", axis = 1, inplace = True)


# In[101]:


train.head(90)


# In[102]:


def fill_avg(student):
    mark = ((student[2] + student [3] + student [4])/3)
    print (mark)


# In[103]:


train["Average"] = train.apply(fill_avg, axis = 1) 


# In[ ]:





# In[104]:


train ["Average"] = fill_avg(train["ID"])


# In[105]:


train.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




