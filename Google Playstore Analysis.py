#!/usr/bin/env python
# coding: utf-8

# # An analysis of Google Playstore App data

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Read data and describe check the data

# In[2]:


playstore = pd.read_csv('googleplaystore.csv')


# In[3]:


playstore.info()


# In[5]:


playstore.head(5)


# ## Data Cleaning and Manipulation

# ### 
# Drop records with null values

# In[6]:


playstore.dropna(inplace=True)


# ### The column "Size" has sizes in kb as well as Mb. 
# Extract the numeric values from the column and multiply if size is mentioned in Mb.

# In[7]:


print(playstore.Size.value_counts())

def change(Size):
    if 'M' in Size:
        x=Size[:-1]
        x=float(x)*1000
        return x
    elif 'k' in Size:
        x=Size[:-1]
        x=float(x)
        return x
    else: return None
    
playstore.Size = playstore.Size.map(change);playstore.Size.value_counts()


# > Convert "Reviews" to a numeric column

# In[8]:


playstore['Reviews']=pd.to_numeric(playstore['Reviews'])


# > Convert "Installs" to integer

# In[9]:


playstore['Installs'] = playstore['Installs'].str.replace("+","",regex=True)
playstore['Installs'] = playstore['Installs'].str.replace(",","",regex=True)

playstore['Installs'] = pd.to_numeric(playstore['Installs'])


#  > Convert the Price column to a numeric field

# In[10]:


playstore['Price']= playstore['Price'].str.replace("$","",regex=True)

playstore['Price'] = pd.to_numeric(playstore['Price'])


# In[12]:


playstore.describe()


# > If reviews exceed the installs, drop.

# In[13]:


playstore.drop(playstore.index[playstore['Reviews']>= playstore['Installs']],inplace=True)


# > Drop rows, if app is free and price is not equal to zero.

# In[14]:


playstore.drop(playstore.index[(playstore['Type']== "Free") & (playstore['Price']> 0)],inplace=True)


# ## EXPLORATORY DATA ANALYSIS

# In[16]:


from matplotlib import style
style.use('ggplot')


# > Histogram of Ratings

# In[20]:


plt.figure(figsize = (12,6))
plt.hist(playstore.Rating,color='teal',edgecolor='black')
plt.title("Ratings")
plt.show()


# > A histogram of size

# In[21]:


plt.figure(figsize = (12,6))
plt.hist(playstore.Size,color='teal',edgecolor='black')
plt.title("Size")
plt.show()


# In[39]:


plt.figure(figsize = (12,6))
sns.boxplot(data=playstore,y='Price')
plt.show()


# In[40]:


plt.figure(figsize = (12,6))
sns.boxplot(data=playstore,y='Reviews')
plt.show()


# ### Treatment for Outliers
# 
# - > Remove rows with very high prices
# - > Remove apps with very few reviews
# - > Remove outliers from the install column

# In[30]:


print("Apps with Prices greater than 50:" +str(len(playstore[playstore.Price>50])))

playstore = playstore[playstore.Price < 50].copy()

print("There are " + str(playstore.shape[0])+ " Rows" + " ," + str(playstore.shape[-1]) +" Columns in the data." )


# In[32]:


print("Apps with more than a million reviews:" +str(len(playstore[playstore.Reviews>=1000000])))

playstore = playstore[playstore.Reviews <= 1000000].copy()

print("There are " + str(playstore.shape[0])+ " Rows" + " ," + str(playstore.shape[-1]) +" Columns in the data." )


# In[35]:


print(playstore.Installs.quantile([.25,.50,.75,.90,.99]))

print("Apps with more than a million installs:" +str(len(playstore[playstore.Installs>=1000000])))

playstore = playstore[playstore.Installs <= 1000000].copy()

print("There are " + str(playstore.shape[0])+ " Rows" + " ," + str(playstore.shape[-1]) +" Columns in the data." )


# #### Scatter plot of Ratings vs Price

# In[44]:


plt.figure(figsize=(12,6))
plt.title('Ratings vs Price')
sns.scatterplot(data=playstore,x='Rating',y='Price',hue='Rating')
plt.show()


# > #### *Ratings increases as Price increases*

# In[45]:


plt.figure(figsize=(12,6))
plt.title('Ratings vs Size')
sns.scatterplot(data=playstore,x='Rating',y='Size',hue='Rating')
plt.show()


# > #### *Largers apps are rated better than smaller apps*

# In[47]:


plt.figure(figsize=(12,6))
plt.title('Ratings vs Reviews')
sns.scatterplot(data=playstore,x='Rating',y='Reviews',hue='Rating')
plt.show()


# > #### *Apps that are rated higher have more reviews.*

# In[48]:


plt.figure(figsize=(12,6))
plt.title('Content Ratings vs Ratings')
sns.boxplot(data=playstore,x='Content Rating',y='Rating',color = 'teal')
plt.show()


# > #### *Adult only apps are rated higher on average when compared with other categories.*

# In[52]:


plt.figure(figsize=(15,6))
plt.title('Categories vs Ratings')
sns.boxplot(data=playstore,x='Category',y='Rating',color = 'teal')
plt.xticks(fontsize = 11,rotation = 'vertical')
plt.show()


# > #### *Apps around books and references seem to have the highest median.*

# # Machine Learning

# ## Data Preprocessing
# - Create a copy of the data and use the copy for machine learning
# - Apply log transformation to the Installs and Review columns
# - Drop unneccesary columns
# - Train and Test the data
# - Build a model

# In[53]:


#create a copy of the dataframe
appdata = playstore.copy()


# In[54]:


#applying log transformation to normalise the data
appdata.Installs = appdata.Installs.apply(func=np.log1p)
appdata.Reviews = appdata.Reviews.apply(func=np.log1p)


# In[55]:


#Drop unneccessary columns
appdata.drop(columns = ['App','Last Updated','Type','Current Ver','Android Ver'],inplace = True)

appdata.head(1)


# In[56]:


data = pd.get_dummies(appdata,columns=['Category','Genres','Content Rating'],drop_first = True)


# In[57]:


data.dropna(inplace=True)


# In[58]:


data


# ### Training and Testing the data

# In[61]:


# define x and y 

X = data.iloc[:,1:]
y = data.Rating


# In[60]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


# In[63]:


from sklearn.linear_model import LinearRegression


# In[64]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[66]:


# Predicting Test Data

predictions = model.predict(X_test)


# ** Create a scatter plot of the actual test values versus the predicted values**

# In[70]:


plt.figure(figsize=(12,6))
plt.scatter(y_test,predictions)
plt.xlabel('Y test')
plt.ylabel('Predicted Y')
plt.show()


# In[71]:


# Evaluating the model

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[88]:


sns.displot((y_test-predictions),bins=50)


# In[89]:


coefficients = pd.DataFrame(model.coef_,X.columns)
coefficients.columns = ['Coefficient']
coefficients


# In[ ]:




