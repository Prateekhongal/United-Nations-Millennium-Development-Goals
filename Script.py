import pandas as pd
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from collections import defaultdict
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from catboost import CatBoostClassifier, Pool
import re
from scipy import stats
import seaborn as sns
sns.set()

import warnings
warnings.filterwarnings("ignore")


test = pd.read_csv('./TrainingSet.csv')
test.shape[0]


test.head()


user = (test.isnull().sum()/test.shape[0]) * 100
user[user>0]


test.dtypes


test


df = pd.read_csv('Your dataset', index_col=[0])
# Remove the [] and everything within from the column names
df.columns = df.columns.str.replace(r"\[.*\]","").str.rstrip()
df.head()


"In total there are {} unique global categories".format(df['Series Code'].str.split('.', expand=True)[0].nunique())


def country_info (country_name):
    """
    param 1 country_name: Country name to filter on.
    returns: A Pandas DataFrame indexed by year with all the series as columns.
    """
    return df.loc[df['Country Name'] == country_name]\
            .set_index('Series Name')\
            .drop(['Country Name', 'Series Code'], axis=1)\
            .T


country_info('India').head()


def series_to_country(series_name):
    """
    param 1 series_name: series_name to filter on.
    returns: A Pandas DataFrame with the country name as columns and the year as index without Series Code and Series Name.
    """
    return df.loc[df['Series Name'] == series_name]\
        .drop(['Series Code', 'Series Name'], axis=1)\
        .set_index('Country Name')\
        .T


series_to_country('Urban population').head(


list_countries = ['Belgium', 'China', 'Tunisia']
plt.subplots(figsize=(15,10))

data = series_to_country('Alternative and nuclear energy (% of total energy use)')[list_countries]
# create the 3 year centered rolling mean of our countries
data['Rolling 3 year'] = data.mean(axis=1).rolling(3, center=True).mean()

plt.xticks(rotation=90)
plt.plot(data.index.values, data)
plt.legend(data.columns.values)
plt.title('Alternative and nuclear energy (% of total energy use)')
plt.ylabel('% of total energy use')
plt.xlabel('year')
plt.show()


plt.subplots(figsize=(15,10))

ele_coal = series_to_country('Electricity production from coal sources (kWh)')

list_countries = ele_coal.mean()\
                           .nlargest(8)\
                           .index.values

ele_coal = ele_coal[list_countries]
plt.xticks(rotation=90)
plt.plot(ele_coal.index.values, ele_coal)
plt.legend(list_countries)
plt.title('The 8 countries with the largest "Electricity production from coal sources (kWh)')
plt.ylabel('kWh')
plt.xlabel('year')
plt.show()


plt.subplots(figsize=(15,10))

df1 = series_to_country('Electricity production from coal sources (kWh)').iloc[-6:]

df1 = df1.loc[:, (df1.iloc[:]>0).all()]
# calculate the % increase between 2002 and 2007 and select the 8 largest indexes
countries = df1.iloc[[0, -1]].pct_change().iloc[1].nlargest(8).index.values
largest_increase = df1[countries]

plt.plot(largest_increase.index.values, largest_increase.pct_change())
plt.legend(largest_increase.columns.values)
plt.title('The 8 countries with the biggest percentual increase in the last 5 years \n that had a nonzero "Electricity production from coal sources (kWh)" in each of the last 5 years of the dataset')
plt.ylabel('% change')
plt.xlabel('year')
plt.show()


df = pd.read_csv('./TrainingSet.csv', index_col=0)
df.columns = [year[:4] for year in df.columns][:-3] + [col.replace(' ', '_') for col in df.columns.values[-3:]]


# read the data containing the rows we need to predict
df_submission = pd.read_csv('./SubmissionRows.csv', index_col=0)


df_submission_in_data = df.loc[df_submission.index]
submission_codes = df_submission_in_data.Series_Code.unique()


def make_prediction(row):
    data = row.loc['1972':'2007']
    nbr_data_points = data.count()
    if nbr_data_points < 2:
        pred_2008 = data.dropna().values
        pred_2012 = pred_2008
    
    else:
        years = data.dropna().index.values.astype(np.int).reshape(-1, 1)
        values = data.dropna().values
        
        # linear regression
        regr = LinearRegression()
        regr.fit(years, values)
        
        # predictions
        pred_2008 = regr.predict(np.array([2008]).reshape(-1, 1))
        pred_2012 = regr.predict(np.array([2012]).reshape(-1, 1))
        
    return pred_2008[0], pred_2012[0]


df_simple_preds = pd.DataFrame(df_submission_in_data.apply(make_prediction, axis=1).tolist(), \
                               index=df_submission_in_data.index, columns=['2008','2012'])


df_simple_preds


df_simple_preds.to_csv("submission.csv")


# Imputation using interpolation.
year_from = 2004

def load_data(year_from):  
    assert isinstance(year_from,int)
    # Imports
    train = pd.read_csv('./TrainingSet.csv',index_col=0)
    submission = pd.read_csv('./SubmissionRows.csv',index_col=0)
    # Remove [YR****] and input '_' for last 4 cols
    train.columns = list(map(lambda x: re.findall(r'\d+',x)[0],train.columns[:36])) + list(map(lambda x: '_'.join(x.split()),train.columns[36:]))
    # Use last 3 years for predictions: This is subjected to change
    train = train.loc[:,f'{year_from}':]
    return train, submission


train, submission = load_data(2004)


def submission_data_missing(train,submission):    
    mask  = train.loc[submission.index,].T.isna().sum()>0
    train_X = train.loc[submission.index,]
    return train_X.loc[mask]


# 16 rows with missing values: Interpolate these values
submission_data_missing(train,submission)


def interpolate_data(train,submission):
    train_X = train.loc[submission.index,:]
    # Interpolate: Both directions are bfilled and ffilled respectively
    train_X.loc[:,:'2007'] = train_X.loc[:,:'2007'].interpolate(limit_direction='both',axis=1)
    return train_X


# Data:
data = interpolate_data(train,submission)

# Func to split that dataframe to values [2004,2005,2006,2007] and [country_name,series_code,series_name]
def split_dataframe(data):
    raw_data = data.loc[:,:'2007']
    description = data.loc[:,'Country_Name':]
    return raw_data,description


# Split:
raw_data,description = split_dataframe(data)


raw_data


description


# Export CSV:
raw_data.to_csv('./raw_data.csv')
description.to_csv('./description.csv')


def format_dataframe(raw_data):
    """
    Quick utility function to format dataframe into a X,y format with X = Numbers of years from the start, y = values.
    Example:
    Initial Dataframe being:
    2005   0.4
    2006   0.6
    2007   0.8
    
    The function transforms it into
    X   y
    0   0.4
    1   0.6
    2   0.8
    
    Returns: X,y
    
    Note: If we have 10 different timeseries (features) X.shape = (n_years,n_features) so slicing will be needed to predict
    individually.
    """
    # Extract index from raw data before transforming:
    raw_data_index = list(raw_data.index)
    raw_data.columns = raw_data.columns.astype('int')
    # Transponse to have time as index instead of columns
    raw_data = raw_data.T
    X = np.asarray(raw_data.index - raw_data.index[0]).reshape(-1,1)
    y = raw_data.values
    return X,y,raw_data_index


X,y,raw_data_index = format_dataframe(raw_data)


def linear_regression_poly(X,y,degree,year):
    assert isinstance(X,np.ndarray)
    assert isinstance(y,np.ndarray)
    assert isinstance(degree,int) and degree > 0
    assert year >= 2004
    year_pred = np.array([[year % year_from]])
    pipe = Pipeline([('poly',PolynomialFeatures(degree=degree)),('SVR',LinearRegression())])
    n_features = y.shape[1]
    predictions_year = defaultdict(list)
    # Fit:
    for i in range(n_features):
        # slice each series:
        y_i = y[:,i]
        pipe.fit(X,y_i)
        # prediction value for year specified
        y_pred = pipe.predict(year_pred)[0]
        predictions_year[f'{year}[YR{year}]'].append(y_pred)
    
     # To dataframe: with correctly indexed submission values
    df = pd.DataFrame(predictions_year,index=raw_data_index)
    return df


# Predictions: Polynomial degree = 1:
_2008 = linear_regression_poly(X,y,1,2008).values
_2012 = linear_regression_poly(X,y,1,2012).values

# Into submission:
submission['2008 [YR2008]'] = _2008
submission['2012 [YR2012]'] = _2012


submission

submission.to_csv("submission.csv")




















