#!/usr/bin/env python
# coding: utf-8

# In[7]:


import nltk


# In[8]:


import pandas as pd


# In[9]:


fake =pd.read_csv("Fake.csv")
gen=pd.read_csv("True.csv")


# In[10]:


display(gen.info())


# In[11]:


fake['target']=0
gen['target']=1


# In[12]:


data=pd.concat([fake,gen],axis=0)


# In[13]:


date=data.reset_index(drop= True)


# In[14]:


data=data.drop(['subject','date','title'],axis=1)


# In[15]:


print(data.columns)


# ## tockenize
# 

# In[16]:


from nltk.tokenize import word_tokenize


# In[17]:


data['text']=data['text'].apply(word_tokenize)


# In[18]:


print(data.head(10))


# ## stemmer

# In[19]:


from nltk.stem.snowball import SnowballStemmer
porter= SnowballStemmer("english")


# In[20]:


def stem_it (text):
    return [porter.stem(word) for word in text]


# In[21]:


data['text']=data['text'].apply(stem_it)


# In[22]:


print(data.head(10))


# ## stopwoed removal

# In[23]:


def stem_it(t):
    dt=[word for word in t if len(word)>2]
    return dt


# In[24]:


data['text']=data['text'].apply(stem_it)


# In[25]:


print(data.head(10))


# In[26]:


data['text']=data['text'].apply(' '.join)


# ## spitting

# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(data['text'],data['target'])
display(X_train.head())
print('\n')
display(Y_train.head())


# 

# ## veactorization

# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
my_tifdf=TfidfVectorizer(max_df=0.7)
tifdf_train= my_tifdf.fit_transform(X_train)
tifdf_test=my_tifdf.transform(X_test)


# In[42]:


print(tifdf_train)


# ## Logic Regression

# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[44]:


model1=LogisticRegression(max_iter=900)
model1.fit(tifdf_train,Y_train)
pred1 =model1.predict(tifdf_test)
Cr1   = accuracy_score(Y_test,pred1)
print(Cr1*100)


# ## PassiveAggressiveClasifier

# In[47]:


from sklearn.linear_model import PassiveAggressiveClassifier

model1 = PassiveAggressiveClassifier(max_iter=50)
model1.fit(tifdf_train,Y_train)


# In[51]:


y_pred= model1.predict(tifdf_test)
accscore = accuracy_score(Y_test,y_pred)
print('The accuracy is',accscore*100)


# In[ ]:




