# from config import db_password
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData
from sqlalchemy import *
import psycopg2
from sqlalchemy.orm import create_session
import pandas as pd 

db_password='Kl@v5SQL'
user_name='Crypto_Team'
aws_password='Cryptocurrency_Project'

url='cryptodb.crgu064gyupd.us-east-2.rds.amazonaws.com'
aws_string=f"postgresql://{user_name}:{aws_password}@{url}:5432/postgres"
engine = create_engine(aws_string)
#Create and engine and get the metadata
Base = declarative_base()
metadata = MetaData(bind=engine)

#reflect table
btc = Table('all_coins_data', metadata, autoload=True, autoload_with=engine)
#Create a session to use the tables    
session = create_session(bind=engine)

#Query database
coin_list = session.query(btc).all()
coin_df=pd.DataFrame(coin_list)
coin_df.to_csv('coin_all.csv')
