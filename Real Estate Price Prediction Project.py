#!/usr/bin/env python
# coding: utf-8

# # Data cleaning

# In[3]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[ ]:





# In[4]:


df1=pd.read_csv("Documents/Datasets/Bengaluru_House_Data.csv")
print(df1)


# In[5]:


df1.head()


# In[6]:


df1.shape


# In[7]:


#inorder to showcase the area types
df1.groupby('area_type')['area_type'].agg('count')


# In[8]:


df2=df1.drop(['area_type','society','balcony','availability'],axis='columns')


# In[9]:


df2.head()


# In[10]:


#inorder to work with dataset we have remove the null value
df2.isnull().sum()


# In[11]:


df2['bath'].unique()


# In[12]:


df3=df2.dropna()


# In[13]:


df3.isnull().sum()


# In[14]:


#here after dropping nan values
#we have ananlyze the data again
df3['size'].unique()


# In[15]:


df3['bhk']=df3['size'].apply(lambda x:int(x.split(' ')[0]))


# In[16]:


df3.head()


# In[17]:


df3['bhk'].unique()


# In[18]:


df3[df3.bhk>20]


# In[19]:


df3.total_sqft.unique()


# In[20]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[21]:


df3[~df3['total_sqft'].apply(is_float)].head(10)


# In[ ]:





# In[22]:


def convert_sqft_to_num(x):
    tokens=x.split('-')
    if(len(tokens)==2):
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None
    


# In[23]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head()


# In[24]:


df4.loc[30]


# # Feature Engineering

# In[25]:


#feature engineering creates new features which helps in further for outlier detection
df5=df4.copy()
df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head()


# In[26]:


#we will explore location column
len(df5['location'].unique())
#here if we do encoding here ,it agian create 1304 new column which agian take a lot of date and time


# In[27]:


df5.location=df5.location.apply(lambda x:x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
print(location_stats)


# In[28]:


#here any location with lessthan 10 location mark it as other
len(location_stats[location_stats<=10])


# In[29]:


location_stats_less_than_10=location_stats[location_stats<=10]
location_stats_less_than_10


# In[30]:


len(df5.location.unique())


# In[31]:


#transoformation
df5.location=df5.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# # outlier removal
# 

# In[32]:


df5.head(10)


# In[33]:


df5[df5.total_sqft/df5.bhk<300].head()
#the below are unusal bedrooms with total sqft
#these are certain kinds of outliers,so we need to remove them
#anamoloyies


# In[34]:


df5.shape


# In[35]:


df6=df5[~(df5.total_sqft/df5.bhk<300)]
df6.shape
#we have removed the outlies


# In[36]:


#now lets check the price_per_sqft
df6.price_per_sqft.describe()


# In[44]:


def remove_pps_outliers(df):
    df_out=pd.DataFrame()
    for key,subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df7=remove_pps_outliers(df6)
df7.shape


# In[46]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.location==location) & (df.bhk==2)]
    bhk3 = df[(df.location==location) & (df.bhk==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.total_sqft,bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.total_sqft,bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[47]:


plot_scatter_chart(df7,"Hebbal")

