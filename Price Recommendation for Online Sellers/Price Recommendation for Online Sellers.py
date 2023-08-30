#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and dataset

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


products = pd.read_csv("data.csv", encoding='unicode_escape')


# In[3]:


products


# In[4]:


#Statistical analysis of each features of the dataset
products.describe(include="all")


# # Data Preprocessing

# In[5]:


#Checking the missing values in the dataset since it can introduce bias and can reduce the performance
fig = px.bar(x=products.columns , y = (products.isnull().sum()/products.shape[0])*100)
fig.show()


# In[6]:


#Dropped this column since it was non informative and unique to each customer
products.drop("CustomerID" , axis = 1 , inplace = True)


# In[7]:


#Null values in this colum was less than 5 % so it was better to drop those
products.dropna(subset=['Description'] , inplace=True)


# In[8]:


#After imputation work , no null values
products.isnull().sum()


# In[9]:


print("Shape of the dataset:" ,products.shape)


# In[10]:


#Checking the datatypes to ensure proper work flow
products.dtypes


# In[11]:


products.head()


# In[12]:


#Formatting the raw date 
products['Date'] = pd.to_datetime(products['InvoiceDate'])
products['Month-Year'] = products['Date'].dt.strftime('%b-%Y')
products.drop(['InvoiceDate','Date'],axis=1,inplace=True)


# In[13]:


products


# # EDA

# In[14]:


#Analysing products with high sales
products["Description"].value_counts()


# In[15]:


#Checking the entities of the max sale product
products_maximum = products[products["Description"] == "WHITE HANGING HEART T-LIGHT HOLDER" ]
products_maximum


# In[16]:


#Now lets even check the entities of 2nd highest sale product
products_sec_maximum = products[products["Description"] == "REGENCY CAKESTAND 3 TIER" ]
products_sec_maximum


# In[17]:


#We can se that even the quantities are varying , so lets check according to the quantity
products_q = products.groupby('Description')['Quantity'].sum().reset_index()
products_q.columns = ['Description', 'Total Quantity']
products_q


# In[18]:


Top_10 = products_q.sort_values(by='Total Quantity', ascending=False).head(10)
Top_10


# In[19]:


#Top 10 items 
plt.figure(figsize=(15, 10))
sns.barplot(data=Top_10, x="Total Quantity", y="Description", capsize=3, palette="winter")
plt.title("Top 10 Items Sold by Overall Quantity")
plt.xlabel("Overall Quantity")
plt.ylabel("Description")
plt.show()


# In[20]:


#Lets extract top 15 products based on the unit price 
products_expensive = products.sort_values(by = "UnitPrice" , ascending=False).head(15)
products_expensive


# In[21]:


# We can see that all the expensive products quantity is -1 , also means they are bought rare


# In[22]:


#Lets look to the top 5 countries 
country_counts = products['Country'].value_counts()

# Select the top 5 countries
top_5_countries = country_counts.head()

# Convert the top countries data into a DataFrame
top_5_df = top_5_countries.reset_index()
top_5_df.columns = ['country', 'count']

# Create a bar plot using Plotly Express
fig = px.bar(top_5_df, x='country', y='count', title='Top 5 Countries')

# Show the plot
fig.show()


# In[23]:


#Visualizing number of sales based on months
products['month'] = products["Month-Year"].str[:3]
month_count_value = products["month"].value_counts()
month_count_value=month_count_value.sort_values()
fig = px.bar(x=month_count_value.index , y = month_count_value.values)
fig.show()


# In[24]:


#Visualizing number of sales based on years
products['year'] = products["Month-Year"].str[4:]
year_count_value = products["year"].value_counts()
year_count_value=year_count_value.sort_values()


fig = px.bar(x=year_count_value.index , y = year_count_value.values)
fig.show()




# In[25]:


#Removing non informative features
products.drop("Description" , axis=1, inplace=True)
products.drop("StockCode" , axis=1, inplace=True)
products.drop("InvoiceNo" , axis=1, inplace=True)
products.drop("Month-Year" , axis=1, inplace=True)
products.drop("year" , axis=1, inplace=True)
products.drop("month" , axis=1, inplace=True)


# In[26]:


products


# In[27]:


#Encoding Categorical column to Numerical column
categorical_cols = ["Country"]


# In[28]:


from sklearn.preprocessing import LabelEncoder
Lc = LabelEncoder()
products["Country"]= Lc.fit_transform(products["Country"])


# In[29]:


products


# In[30]:


#Adding total price column
products['Total Price'] = products['UnitPrice'] * products['Quantity']
products.head()


# In[31]:


#Analysing Correlation
sns.heatmap(products.corr(), cmap='Purples', annot=True, fmt=".2f")


# # Data spliting and Modelling

# In[32]:


#Splitting the dataset
X = products.drop("Total Price" , axis =1)

#Since we need to predict total price only so let make it our target dependent variable

y = products["Total Price"]


# In[33]:


X


# In[34]:


y


# In[35]:


#Train test split
from sklearn.model_selection import train_test_split


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[37]:


#Modelling Part
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# In[38]:


#Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[39]:


models = [
    LinearRegression(),
    RandomForestRegressor(n_estimators=104, random_state=42),
    Lasso(alpha = 10),
    Ridge(alpha = 10)
]


# # Model Selection

# In[40]:


for mo in models:
    mo.fit(X_train, y_train)
    y_pred = mo.predict(X_test)
    
    r2_value = r2_score(y_test,y_pred)
    print(f"{mo.__class__.__name__} R2 Score: {r2_value}")


# <div class="alert alert-block alert-info""> Insights: 
#                                           
# Now our model is ready after being trained we have used 4 regressor out of which r2 lasso is giving the best result currently.So we will go with that now</div> 

# In[41]:


r2_lasso = []


# In[42]:


for x in range(10,500,10):
    model_l = Lasso(alpha = x)
    model_l.fit(X_train, y_train)
    y_pred = model_l.predict(X_test)

    r2_value = r2_score(y_test,y_pred)
    r2_lasso.append(r2_value)


# In[43]:


#Varios results based on changing the hyperparameter
r2_lasso


# In[44]:


sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))
sns.lineplot(x=list(range(10,400,10)), y=r2_lasso[:39], marker='o', color='blue', linestyle = 'dashed')
plt.title('R2 Score variation for Lasso Regression Model',fontsize=14)
plt.xlabel('Alpha value',fontsize=14)
plt.ylabel('R2 Score',fontsize=14)
plt.show()


# In[45]:


max(r2_lasso)


# <div class="alert alert-block alert-info""> 
#                                           
# Our model is completed and ready to deploy , the regressor selection can be done based on user reequirement and type of the dataset</div> 

# In[ ]:




