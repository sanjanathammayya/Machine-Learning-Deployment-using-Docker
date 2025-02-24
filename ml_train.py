#!/usr/bin/env python
# coding: utf-8

# ##  Predict  Customer Behavior
# 
# Binary Classification problem 

# In[1]:


# Import libraries
import numpy as np
import pandas as pd

# ### import data

# The dataset that we have in this case is from an online platform about the historical transactions of customers. It contains the data points such as age, total pages viewed, and whether the customer is a new or repeat customer. The output variable contains whether the customer bought the product online or not.

# In[2]:


#read the data
df = pd.read_csv('online_sales.csv')

# In[3]:


df.shape

# In[4]:


df.head()

# In[5]:


#target class frequency
df.converted.value_counts()

# We can clearly see there is a skewed target class in this dataset that typically needs to be treated by some undersampling/oversampling technique

# In[6]:


df.info()

# In[7]:


df.describe()

# ### Preparing Data For Modeling

# In[8]:


input_columns = [column for column in df.columns if column != 'converted']
output_column = 'converted'
print (input_columns)
print (output_column)

# In[9]:


#input data
X = df.loc[:,input_columns].values
#output data 
y = df.loc[:,output_column]
#shape of input and output dataset
print (X.shape, y.shape)

# **In ideal ML scenarios, proper data exploration and feature engineering are advised before model training, Since the overall idea is to deploy the ML app, the focus is on the containerizing the app instead of improving the accuracy of the model**

# ### Modeling : Logistic Regression

# we are going to train a simple logistic regression model to make the predictions on the test data and later export it for deployment purposes

# In[10]:


#import model specific libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# In[11]:


#Split the data into training and test data (70/30 ratio)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=555, stratify=y)

# In[12]:


#validate the shape of train and test dataset
print (X_train.shape)
print (y_train.shape)

print (X_test.shape)
print (y_test.shape)

# In[13]:


#check on number of positive classes in train and test data set
print(np.sum(y_train))
print(np.sum(y_test))

# ## Train the Logistic Model

# In[14]:


#fit the logisitc regression model on training dataset 
logreg = LogisticRegression(class_weight='balanced').fit(X_train,y_train)

# In[15]:


logreg.score(X_train, y_train)

# In[16]:


#validate the model performance on unseen data
logreg.score(X_test, y_test)

# In[17]:


#make predictions on unseen data
predictions=logreg.predict(X_test)

# ## Results 

# In[18]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions,target_names=["Non Converted", "Converted"]))

# In[19]:


logreg

# ## Export Model 

# In[20]:


### Create a Pickle file using serialization 
import pickle

pickle_out = open("logreg.pkl","wb")
pickle.dump(logreg, pickle_out)
pickle_out.close()

# In[21]:


pickle_in = open("logreg.pkl","rb")
model = pickle.load(pickle_in)

# In[22]:


model

# In[23]:


#predict using the model on customer input
model.predict([[32,1,1]])[0]


# In[24]:


#Group prediction (multiple customers)
df_test = pd.read_csv('test_data.csv')
predictions = model.predict(df_test)

print(list(predictions))

# As we can observe, the model seems to be making predictions for a single customer as well as a group of customers. 
# 
# Now we can move on to the next step of building a Flask app to run this model.
