#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from nltk.tokenize import word_tokenize,sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
import string
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing



# In[7]:


import nltk
nltk.download('stopwords')


# In[10]:


data = pd.read_csv('twitter_training.csv')
v_data = pd.read_csv('twitter_validation.csv')


# In[9]:


data


# In[11]:


v_data


# In[12]:


data.columns=['id','game','sentiment','text']
v_data.columns=['id','game','sentiment','text']


# In[13]:


data


# In[14]:


v_data


# In[15]:


data.shape


# In[16]:


data.columns


# In[17]:


data.describe(include='all')


# In[19]:


id_types = data['id'].value_counts()


# In[20]:


id_types


# In[21]:


plt.figure(figsize=(12,7))
sns.barplot(y=id_types.index,x=id_types.values)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of id vs Count')
plt.show()


# In[22]:


game_types = data['game'].value_counts()
game_types


# In[23]:


plt.figure(figsize=(14,10))
sns.barplot(x=game_types.values,y=game_types.index)
plt.title('# of Games and their count')
plt.ylabel('Type')
plt.xlabel('Count')
plt.show()


# In[24]:


sns.catplot(x="game",hue="sentiment",kind="count",height=10,aspect=3, data=data)


# In[26]:


sns.heatmap(data.isnull(),yticklabels=False,cmap='viridis')


# In[34]:


data.dropna(subset=['text'],inplace=True)
total_null=data.isnull().sum().sort_values(ascending=False)
percent=((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending = False)

print("Total records =",data.shape[0])
missing_data = pd.concat([total_null,percent.round(2)],axis=1,keys=['Total Missing','In Percent'])
missing_data.head()


# In[35]:


train0=data[data['sentiment']=="Negative"]
train1=data[data['sentiment']=="Positive"]
train2=data[data['sentiment']=="Irrelevant"]
train3=data[data['sentiment']=="Neutral"]


# In[37]:


train0.shape, train1.shape, train2.shape, train3.shape


# In[38]:


data = pd.concat([train0,train1,train2,train3],axis=0)
data


# In[39]:


id_types = data['id'].value_counts()
id_types


# In[40]:


plt.figure(figsize=(12,7))
sns.barplot(x=id_types.values,y=id_types.index)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of Tv Shows vs Movies')
plt.show()


# In[41]:


game_types = data['game'].value_counts()
game_types


# In[43]:


plt.figure(figsize=(20,20))
sns.barplot(x=game_types.values,y=game_types.index)
plt.xlabel('Type')
plt.ylabel('Count')
plt.title('# of TV shows vs Movies')
plt.show()


# In[44]:


sentiment_types = data['sentiment'].value_counts()
sentiment_types


# In[46]:


plt.figure(figsize=(12,7))
plt.pie(x=sentiment_types.values,labels=sentiment_types.index,autopct='%.1f%%', explode=[0.1,0.1,0,0])
plt.title('The difference in the type of contents')
plt.show()


# In[47]:


sns.catplot(x='game',hue='sentiment',kind='count',height=7,aspect=2,data=data)


# In[48]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[49]:


data['sentiment']=label_encoder.fit_transform(data['sentiment'])
data['game']=label_encoder.fit_transform(data['game'])
v_data['sentiment']=label_encoder.fit_transform(v_data['sentiment'])
v_data['game']=label_encoder.fit_transform(v_data['game'])


# In[50]:


data=data.drop(['id'],axis=1)
data


# In[51]:


data.nunique()


# In[52]:


v_data.nunique()


# In[ ]:




