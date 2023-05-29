#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


# In[19]:


# features
#pelo longo
#perna curta
#faz auau
porco1=[0,1,0]
porco2=[0,1,1]
porco3=[1,1,0] 

cachorro1=[0,1,1]
cachorro2=[1,0,1]
cachorro3=[1,1,1]

train_x = [porco1, porco2,porco3, cachorro1,cachorro2,cachorro3]
train_y = [1,1,1,0,0,0]


# In[20]:


model = LinearSVC()
model.fit(train_x, train_y)


# In[21]:


animal_misterioso=[1,1,1]

model.predict([animal_misterioso])


# In[22]:


misterio1= [1,1,1]
misterio2= [1,1,0]
misterio3= [0,1,1]

test_x  = [misterio1, misterio2, misterio3]
test_y = [0,1,1]

previsoes = model.predict(test_x)


# In[32]:


taxa_de_acerto = accuracy_score(test_y, previsoes)

print("Taxa de acerto: %.2f" % (taxa_de_acerto*100), "%")

