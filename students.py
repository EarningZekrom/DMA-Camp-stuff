#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[135]:


train = pd.read_csv("StudentsPerformance.csv")


# In[136]:


print (train)


# In[137]:


train.head (10)


# In[138]:


train.drop ("gender", axis = 1, inplace = True)


# In[139]:


train.head()


# In[140]:


train.drop ("race/ethnicity", axis = 1, inplace = True)


# In[141]:


train.drop ("parental level of education", axis = 1, inplace = True)


# In[142]:


train.head(90)


# In[143]:


def fill_avg(student):
    mark = ((student[2] + student [3] + student [4])/3)
    print (mark)


# In[167]:


train["Average"] = train.apply(fill_avg, axis = 1) 


# In[166]:





# In[159]:


train ["Average"] = fill_avg


# In[160]:


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




