#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing Basic Libraries
import pandas as pd
import numpy as np


# In[2]:


#Reading the dataset
df = pd.read_csv("telecom_churn.csv")


# In[3]:


df


# In[4]:


#Checking Null values
df.isnull().sum()


# In[5]:


df.info()


# In[6]:


#Statistical analysis of the features
df.describe(include="all")


# In[7]:


df.columns


# In[8]:


#Gender class numbers
df["gender"].value_counts()


# In[9]:


df.shape


# In[10]:


#Dropped useless column
df.drop(["customer_id"] ,axis=1, inplace=True)


# In[11]:


df


# In[12]:


#Importing lib for eda
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px


# In[13]:


df["age"].value_counts().head()


# In[14]:


import plotly.subplots as sp


# In[15]:


#Count plot for each feature
# Create a subplot layout
fig = sp.make_subplots(rows=5, cols=3, subplot_titles=df.columns[:-1], shared_xaxes=True, shared_yaxes=True)

# Populating the subplots with histograms
for i, col in enumerate(df.columns[:-1]):
    trace = go.Histogram(x=df[col], name=col)
    fig.add_trace(trace, row=(i // 3) + 1, col=(i % 3) + 1)

# Update axis labels and titles
for i, col in enumerate(df.columns[:-1]):
    fig.update_xaxes(title_text=col, row=(i // 3) + 1, col=(i % 3) + 1)
    fig.update_yaxes(title_text='Count', row=(i // 3) + 1, col=(i % 3) + 1)

# Update layout
fig.update_layout(title='Histogram Subplots', height=800)


fig.show()


# In[16]:


#Age distribution
fig = px.histogram(df, x='age', title='Age Distribution Histogram')

# Update axis labels
fig.update_layout(xaxis_title='Age', yaxis_title='Count')

# Show the plot
fig.show()


# In[17]:


age_bins = [i for i in range(1, 101, 10)]


# In[18]:


df['age_group'] = pd.cut(df['age'], bins=age_bins)


# In[19]:


df


# In[20]:


age_group_counts = df['age_group'].value_counts().sort_index()


# In[21]:


age_group_counts


# In[22]:


df_ag = pd.DataFrame(age_group_counts)
df_ag


# In[23]:


fig = px.bar(x=[i for i in range(0,9)], y=df_ag['age_group'], labels={'x': 'Age Group', 'y': 'Count'})


# In[24]:


#Age group wise analysis
fig.update_xaxes(tickvals=age_bins)

# Update layout
fig.update_layout(title='Age Group Count Bar Plot')


# In[25]:


category_counts = df['telecom_partner'].value_counts()
sum = 0
for i in range(0,4):
    sum = sum + category_counts[i]
sum


# In[26]:


fig = px.pie(values=(category_counts.values / sum), names=category_counts.index, title='Category Pie Chart')


# In[27]:


#Company wise distribution
fig.show()


# In[28]:


df["telecom_partner"].value_counts()


# In[29]:


#Presenting top 10 state
top_10_state = df["state"].value_counts().sort_values(ascending=False).head(10)


# In[30]:


fig = px.bar(x=top_10_state.index , y = top_10_state.values , labels = {"x":"Top 10 States" , "y" : "Counts"} , title='Top 10 States')


# In[31]:


fig.show()


# In[32]:


gender = df["gender"].value_counts()

fig = px.pie(values=gender.values , names=gender.index , title = "Gender wise distribution")
fig.show()


# In[33]:


#Statistics of Data used feature
data_used=df["data_used"].describe()
fig = px.bar(x=data_used.values[1:] , y = data_used.index[1:] , title = "Data used Statical summary")
fig.show()


# In[34]:


churn_data_data_used=df[df['churn']==1]['data_used']
non_churn_data_used=df[df['churn']==0]['data_used']


# In[35]:


churn_value_calls_made=list(churn_data_data_used.value_counts().sort_values())
non_churn_value_calls_made=list(non_churn_data_used.value_counts().sort_values())
plt.figure(figsize=(10,5))
sns.scatterplot(churn_value_calls_made,label="churn")
sns.scatterplot(non_churn_value_calls_made,label="non churn")
plt.xlabel("Data used")
plt.title("scatterplot data usage made churn  and non-churn customer")


# In[36]:


df = df.drop("age_group" , axis =1)


# In[37]:


#Heatmap
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True)
plt.title("correlation map")
plt.show()


# In[38]:


#Label encoding categorical columns to numerical
from sklearn.preprocessing import LabelEncoder
model=LabelEncoder()
for i in df.columns:
    if df[i].dtype=="object":
        df[i]=model.fit_transform(df[i])


# In[39]:


df


# In[40]:


#Train Test Split
from sklearn.model_selection import train_test_split
X=df.drop("churn",axis=1)
y=df['churn']


# In[41]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)


# In[42]:


#Feature scaling
from sklearn.preprocessing import StandardScaler

model=StandardScaler()

X_train=pd.DataFrame(model.fit_transform(X_train),columns=X_train.columns)
X_test=pd.DataFrame(model.fit_transform(X_test),columns=X_test.columns)


# In[43]:


#Initialising the model
from sklearn.linear_model import LogisticRegression

Model = LogisticRegression()

#Fitting to Logistic regression model
Model.fit(X_train,y_train)


# In[44]:


y_pred = Model.predict(X_test)


# In[45]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[46]:


#Evaluating the result
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:




