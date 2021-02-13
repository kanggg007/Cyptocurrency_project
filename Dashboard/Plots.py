#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# data=yf.download('BTC-USD',start='2020-01-21', interval='1h',  end='2021-01-21',progress=False)[['Close']]
# data.head()
 
# data.plot(figsize=(10,10))


# In[2]:


import pandas as pd  
from sqlalchemy import create_engine 
import mysql.connector as mysqlConnector
from config2 import db_password

db_string = f"postgres://postgres:{db_password}@localhost/cryptocurrency_db"
engine = create_engine(db_string)

# SQLAlchemy connectable 
cnx = create_engine('sqlite:///cryptocurrency_db').connect()

li = []
coin_list = [
'BTC','ETH','USDT','DOT','XRP',
'ADA','LINK','LTC','BCH','XLM',
'BNB','DOGE','USDC','HEX','UNI',
'WBTC','AAVE','BSV','EOS','CEL'
]

for coin in coin_list:
    df = pd.read_sql_table(coin + '_data', con=engine)
    li.append(df)
    

df = pd.concat(li, ignore_index=True)
df = df.fillna(0)

df

# table named 'contacts' will be returned as a dataframe. 
# df = pd.read_sql_table('ADA_data', cnx) 
# print(df)


# In[3]:


df['asset_id']=df['asset_id'].astype(str)


# In[4]:



id_df = pd.read_sql_table('crypto_id', con=engine)
id_df['asset_id']=id_df['asset_id'].astype(str)

print(id_df)


# In[5]:


df= pd.merge(df, id_df, how="left", on=["asset_id", "asset_id"])
# df.to_sql(name= 'all_data', con=engine)


# In[6]:


df


# In[7]:


plt.figure(figsize=(200,80))
fig,ax=plt.subplots()
y_axis = df.loc[(df['asset_id']=='1'),['close']]
y_axis2 = df.loc[(df['asset_id']=='2'),['close']]
y_axis3 = df.loc[(df['asset_id']=='7'),['close']]
y_axis4 = df.loc[(df['asset_id']=='2780'),['close']]
y_axis5 = df.loc[(df['asset_id']=='3'),['close']]
y_axis6 = df.loc[(df['asset_id']=='11'),['close']]
y_axis7 = df.loc[(df['asset_id']=='18'),['close']]
y_axis8 = df.loc[(df['asset_id']=='4'),['close']]
y_axis9 = df.loc[(df['asset_id']=='5'),['close']]
y_axis10 = df.loc[(df['asset_id']=='10'),['close']]

x_axis = df.loc[(df['asset_id']=='1'),['time']]
x_axis2 = df.loc[(df['asset_id']=='2'),['time']]
x_axis3 = df.loc[(df['asset_id']=='7'),['time']]
x_axis4 = df.loc[(df['asset_id']=='2780'),['time']]
x_axis5 = df.loc[(df['asset_id']=='3'),['time']]
x_axis6 = df.loc[(df['asset_id']=='11'),['time']]
x_axis7 = df.loc[(df['asset_id']=='18'),['time']]
x_axis8 = df.loc[(df['asset_id']=='4'),['time']]
x_axis9 = df.loc[(df['asset_id']=='5'),['time']]
x_axis10 = df.loc[(df['asset_id']=='10'),['time']]


plt.plot(x_axis, y_axis, label='BTC', color="red")
plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Cryptocurries Price")
ax2=ax.twinx()
plt.plot(x_axis2, y_axis2, label='ETH')
plt.plot(x_axis3, y_axis3, label='USDT')
plt.plot(x_axis4, y_axis4, label='DOT')
plt.plot(x_axis5, y_axis5, label='XRP')
plt.plot(x_axis6, y_axis6, label='ADA')
plt.plot(x_axis7, y_axis7, label='LINK')
plt.plot(x_axis8, y_axis8, label='LTC')
plt.plot(x_axis9, y_axis9, label='XLM')
plt.plot(x_axis10, y_axis10, label='BNB')
plt.legend()



# In[8]:



coin_df = df.loc[(df['asset_id']=='2')]
plt.figure(figsize=(12,8))
fig,ax=plt.subplots()
y_axis = coin_df['reddit_posts']
y_axis2 = coin_df['close']
x_axis = coin_df['time']
plt.plot(x_axis, y_axis, label='Reddit Posts', color="red")
plt.xlabel("Date")
plt.ylabel("Posts")

plt.title(id_df.loc[1,"symbol"] + " Reddit Posts vs. Price")
ax2=ax.twinx()
plt.plot(x_axis, y_axis2, label='Price')
plt.ylabel('Price')


# In[9]:


df_top_20 = df.loc[df['time'] == '2021-01-25']
df_top_20 = df_top_20.sort_values('market_cap', ascending=False)
# # df_top_20['market_cap']=df_top_20['market_cap'].astype('float64')
# df_top_20['market_cap']=df_top_20['market_cap'].map("{:,}".format)


# In[10]:


df_top_20 = df_top_20.reset_index()
df_top_20


# In[11]:



df_top_20mc = df_top_20[['name','market_cap']]

# df_top_20mc['market_cap']=df_top_20mc['market_cap'].apply(lambda x: '{:.2f}'.format(x))
df_top_20mc


# In[12]:


# Set the x-axis to a list of strings for each month.
x_axis = df_top_20mc['name']

# Set the y-axis to a list of floats as the total fare in US dollars accumulated for each month.
y_axis = df_top_20mc['market_cap']
plt.figure(figsize=(9, 7))
# Create the plot with ax.plt()
# fig, ax = plt.subplots()
plt.bar(x_axis, y_axis, color="blue")
plt.xticks(rotation='vertical')
plt.xlabel("Cryptocurrency")
plt.ylabel("Market Cap (000m)")

plt.title("Top Market Cap cryptocurrencies")


# In[13]:


df_top_20_social = df_top_20[['symbol',
                             'url_shares',
                             'reddit_posts',
                             'tweets',
                             'news',
                             'youtube']]                     


# In[14]:


df_top_20_social


# In[15]:


df_top_20_social['sum'] = df_top_20_social.sum(axis=1)
df_top_20_social['sum']


# In[16]:


df_top_20_social


# In[17]:


# library
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 13))
title = plt.title('Social media posts of selected currency vs total posts')


# create data
x_axis=df_top_20_social['symbol']
y_axis=df_top_20_social['sum']
porcent = 100.*y_axis/y_axis.sum()
# Create a circle for the center of the plot
# central_circle = plt.Circle((0, 0), 0.5, color='white')
# from palettable.colorbrewer.qualitative import Pastel1_7
plt.pie(y_axis, pctdistance=1.1, shadow=True, labeldistance=1.1, wedgeprops = {'linewidth': 3})


centre_circle = plt.Circle((0,0),0.75,color='black', fc='white',linewidth=1.25)
# plt.pie(y_axis, explode=explode_values, labels=x_axis, autopct='%.1f%%')
plt.legend(x_axis, loc=0.0)
fig = plt.gcf()

patches, texts = plt.pie(y_axis, startangle=90, radius=1.2)
labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x_axis, porcent)]

sort_legend = True
if sort_legend:
    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y_axis),
                                          key=lambda x: x[2],
                                          reverse=True))

plt.legend(patches, labels, loc='upper right', bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
fig.gca().add_artist(centre_circle)

plt.subplots_adjust(left=0.0, bottom=0.1, right=0.7)
plt.show()


# In[18]:


df2=df_top_20_social.drop(['sum'], axis=1)
df2[['url_shares','reddit_posts','tweets','news','youtube']]=df2[['url_shares','reddit_posts','tweets','news','youtube']].apply(pd.to_numeric)
df2


# In[19]:


new_max = 100
new_min = 0
new_range = new_max - new_min


# In[20]:


factors = ['url_shares','reddit_posts','tweets','news','youtube']
for factor in factors:
  max_val = df2[factor].max()
  min_val = df2[factor].min()
  val_range = max_val - min_val
  df2[factor + '_Adj'] = df2[factor].apply(
      lambda x: (((x - min_val) * new_range) / val_range) + new_min)
    


# In[21]:


df3 = df2.loc[:, ['symbol', 'url_shares_Adj','reddit_posts_Adj','tweets_Adj','news_Adj','youtube_Adj']]

df3.rename(columns={
    'url_shares_Adj': 'url_shares',
    'reddit_posts_Adj': 'reddit_posts',
    'tweets_Adj': 'tweets',
    'news_Adj': 'news',
    'youtube_Adj': 'youtube'
}, inplace=True)


# In[22]:


df3


# In[32]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import pi

categories=list(df3)[1:]
N = len(categories)
 
# We are going to plot the first line of the data frame.
# But we need to repeat the first value to close the circular graph:
values=df3.loc[1].drop('symbol').values.flatten().tolist()
values += values[:1]
values
 
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]
 
# Initialise the spider plot
ax = plt.subplot(111, polar=True)
 
# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories, color='grey', size=8)
 
# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)
plt.ylim(0,75)
 
# Plot data
ax.plot(angles, values, linewidth=1, linestyle='solid')
 
# Fill area
ax.fill(angles, values, 'b', alpha=0.1)
ax.set_title([df3.loc[1,"symbol"]])


# In[ ]:





# In[ ]:




