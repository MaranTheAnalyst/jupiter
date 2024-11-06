#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dataset=pd.read_csv("heart.csv")
dataset


# In[ ]:


x=dataset.drop(["max_hr","exang","oldpeak","slope","num_major_vessels","thal","target"],axis="columns")
x


# In[ ]:


y=dataset.target
y


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


x_train.shape


# In[ ]:


y_train.shape


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
accuracy=[]
for i in range(1,10):
    model=DecisionTreeClassifier(max_depth=i,random_state=0)
    model.fit(x_train,y_train)
    pred=model.predict(x_test)
    score=accuracy_score(y_test,pred)
    accuracy.append(score)
plt.figure(figsize=(12,6))
plt.plot(range(1,10),accuracy,color="red",marker="o",markerfacecolor="green",linestyle="dashed")
plt.title=("heart")
plt.xlabel("accuracy")
plt.ylabel("level")
plt.show()


# In[ ]:


model=DecisionTreeClassifier(max_depth=4,criterion="entropy")
model.fit(x_train,y_train)


# In[ ]:


pred=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score
print("accuracy of the model precentage:{0}%".format(accuracy_score(pred,y_test)*100))


# In[ ]:


dataset


# In[ ]:


age=int(input("Enter age"))
sex=int(input("Enter gender {male:1},{female:0}"))
chest=int(input("Enter pain type press only 0,1,2,3"))
resting_bp=float(input("Enter bp level"))
cholestoral=float(input("Entger cholestoral level"))
blood_sugar=int(input("Enter blood_sugar press onlt 0 0r 1"))
restecg=input(input("Enter restecg press only 0 or 1"))
heart=[[age,sex,chest,resting_bp,cholestoral,blood_sugar,restecg]]
result=model.predict(heart)
print(result)
if result==1:
    print("heart diseas is confirm pls consult for doctor")
else:
    print("No heart diseas dont worry")








# In[ ]:





# In[ ]:




