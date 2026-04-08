# DIWALI SALES

''' BUSINESS PROBLEM:- A sales company needs to analyze its 
previous year sales data to reduce its cost, 
maximize its sales and focus on those product which the 
people can buy more.

Its objective is to :
Maximize the sales.
Focus on most buying product by advertising.
Increase quantity of  most buying product.

BUSINESS CONSTRAINT:- MInimize cost

'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sqlalchemy import create_engine, text

df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Diwali Sales Data.csv", encoding='latin-1')

# Credentials to connect to Database
from urllib.parse import quote

# Define connection details
user = 'root'
pw = quote('Akshay@123')  # URL encode the password
db = 'DiwaliSalesdb'

# Create connection string with URL-encoded password
conn_str = f"mysql+pymysql://{user}:{pw}@localhost/{db}"

# Create engine
engine = create_engine(conn_str)

# to_sql() - function to push the dataframe onto a SQL table.
df.to_sql('Diwali_Sales_tbl', con = engine, if_exists = 'replace', chunksize = 1000, index = False)



###### To read the data from MySQL Database
sql = 'select * from Diwali_Sales_tbl;'


# for sqlalchemy 1.4.x version (old version)
# df = pd.read_sql_query(engine, sql)

# for sqlalchmey 2.x version (new version)
df = pd.read_sql_query(text(sql), engine.connect())


# First we remove NAN value column using drop command
df = df.drop("Status", axis=1)
df = df.drop("unnamed1",axis=1)

#Head by default give the top 5 values of the dataframe
df.head()

df.shape

df.dtypes

#Typecasting
df.User_ID = df.User_ID.astype('str')

df.info()

df.duplicated().sum()

df1 = df.duplicated()

df = df.drop_duplicates(keep = False) 

df.isna().sum()

sns.boxplot(df.Amount)

df.Amount.mean()
df.Amount.max()
df.Amount.min()

IQR = df['Amount'].quantile(0.75) - df['Amount'].quantile(0.25)

lower_limit = df['Amount'].quantile(0.25) - (1.5 * IQR)
upper_limit = df['Amount'].quantile(0.75) + (1.5 * IQR)

df.Amount.skew()
df.Amount.median()
#--------------------------------------------------------------------------------
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median')
df["Amount"] = pd.DataFrame(imputer.fit_transform(df[["Amount"]]),index=df.index)
df["Amount"].isna().sum()  

df.Amount.median()
#--------------------------------------------------------------------------------
from feature_engine.outliers import Winsorizer
winsor_iqr = Winsorizer(capping_method = 'iqr',                         
                          tail = 'both', 
                          fold = 1.5, 
                          variables = ['Amount'])

df_s= winsor_iqr.fit_transform(df[['Amount']])
sns.boxplot(df_s.Amount)

df["Amount"].median()

# Total amount by sales of top 10 states

sales_state = df.groupby(['State'], as_index = False)['Amount'].sum().sort_values(by = 'Amount', ascending = False).head(10)

sns.barplot(data = sales_state, x = 'State', y = 'Amount')
plt.xticks(rotation = 45)
plt.title('Total amount by sales of top 10 states')
plt.grid(True)
plt.xlabel('State')
plt.ylabel('Amount')


sns.kdeplot(df['Marital_Status'], shade = True, color = 'green')
plt.title('Count Of Marital Status')
plt.xlabel('Marital_Status')
plt.ylabel('Density')  


plt.hist(df['Marital_Status'], color = 'green')
plt.title('Count Of Marital Status')
plt.xlabel('Marital_Status')
plt.ylabel('Frequency') 
df.Marital_Status.value_counts()


sales_state1 = df.groupby(['Marital_Status', 'Gender'], as_index = False)['Amount'].sum().sort_values(by = 'Amount', ascending = False)


sns.barplot(data = sales_state1, y = 'Amount', x = 'Marital_Status', hue = 'Gender')
plt.title('Total sales w.r.t Marital Status and Gender')
plt.xlabel('Marital_Status')
plt.ylabel('Amount') 

sales_state2 = df.groupby(['Occupation'], as_index = False)['Amount'].sum().sort_values(by = 'Amount', ascending = False)

sns.barplot(data = sales_state2, y = 'Amount', x = 'Occupation')
plt.title('Total sales w.r.t Occupation')
plt.xlabel('Occupation')
plt.ylabel('Amount') 
plt.xticks(rotation = 45)


sales_state3 = df.groupby(['Product_Category'], as_index = False)['Amount'].sum().sort_values(by = 'Amount', ascending = False)

sns.barplot(data = sales_state3, y = 'Amount', x = 'Product_Category')
plt.title('Total sales w.r.t Product_Category')
plt.xlabel('Product_Category')
plt.ylabel('Amount') 
plt.xticks(rotation = 90)

