import mysql.connector
from mysql.connector import Error
import pandas as pd
import sqlalchemy as sql


host='localhost'
database='nlpmvpdb'
user='nlpuser'
password='_Robo1980'

def read_df_sqlalchemy(query):    
    try:
        connect_string = 'mysql://{}:{}@127.0.0.1/{}'.format(user, password, database)
        sql_engine = sql.create_engine(connect_string)
        with sql_engine.connect() as connection:
            df = pd.read_sql_query(query, sql_engine)
            return df
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        # sql_engine.close()
        sql_engine.dispose()
        # print("MySQL connection is closed")


# query = "SELECT * FROM nlpmvpdb.user_utterances;"
# df = read_df_sqlalchemy(query)
# print(df.head())
