#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[2]:


uri='https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv'

dados = pd.read_csv(uri)


# In[3]:


x= dados[["home","how_it_works","contact"]]
y= dados["bought"]


# In[22]:


from sklearn.model_selection import train_test_split

SEED = 20
train_x, test_x, train_y, test_y =train_test_split(x,y,
                                                   random_state=SEED,
                                                   stratify=y,
                                                   test_size = 0.25)
modelo = LinearSVC()
modelo.fit(train_x, train_y)
previsoes = modelo.predict(test_x)
taxa_de_acertos =accuracy_score(test_y, previsoes) *100

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(train_x), len(test_x)))
print("A acurácia foi %.2f%%" % taxa_de_acertos)

