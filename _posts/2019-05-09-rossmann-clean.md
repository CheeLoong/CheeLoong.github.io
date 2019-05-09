---
title: "Data Cleaning & Feature Enginnering on Rossmann"
date: 2019-05-09
permalink: /clean-rossmann/
tags: [fastai, feature engineering, data cleaning, rossmann, kaggle]
excerpt: "Preparing the dataset before we build neural net on Rossmann"
mathjax: "true"
---

In this blogpost, we will be using [Rossmann dataset](https://www.kaggle.com/c/rossmann-store-sales/data) which was featured on Kaggle 3 years ago.

It has historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set, but before we go about to do that (which will be in the next blogpost), we will explore feature engineering and data cleaning part of the dataset.

As usual, let's do the cloud server set up before proceeding..

# Colab Cloud Server VM setup & FastAI Configurations

First we need to use GPU for deep learning applications, so instead of buying, we can rent it from cloud servers. There are many other server options, but basically I chose Colab because issa FREE.

To learn how to set up a Colab Server which supports fastai library and its applications, click [here](https://course.fast.ai/start_colab.html).

NB: This is a free service that may not always be available, and requires extra steps to ensure your work is saved. Be sure to read the docs on the Colab web-site to ensure you understand the limitations of the system.


```python
# Permit collaboratory instance to read/write to Google Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
base_dir = root_dir + 'fastai-v3/'
```

    Mounted at /content/gdrive



```python
  !curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.



```python
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```python
from fastai.basics import *
```

# Rossmann

## Getting the Data

First of all, let's try to get the data, note that In addition to the provided data, we will be using external datasets put together by participants in the Kaggle competition. We can download all of them [here](http://files.fast.ai/part2/lesson14/rossmann.tgz).

After we got the the `.tgz` file, we can untar them and upload to which `PATH` is pointing below.


```python
data_dir = Path("/content/data/")
folder = 'rossmann'

PATH = data_dir/folder
PATH.mkdir(parents=True, exist_ok=True)
```

After we have uploaded all the csv files in `PATH`, we will need to read it to Python, the code below shows how we can read multiple csv files as dataframes in Python and in the order we specified.


```python
table_names = ['train', 'store', 'store_states', 'state_names', 'googletrend', 'weather', 'test']
tables = [pd.read_csv(PATH/f'{fname}.csv', low_memory=False) for fname in table_names]
train, store, store_states, state_names, googletrend, weather, test = tables
len(train),len(test)
```




    (1017209, 41088)



## Understanding the Data

### Original Historical Sales Data

There are 3 datasets, `train`, `test`, and `store` are provided in the competition, click [here](https://www.kaggle.com/c/rossmann-store-sales/data) to look at the provided information about the dataset.

Let's first visit `train` which represent the sales data in our train set.


```python
train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>13995</td>
      <td>1498</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>4822</td>
      <td>559</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



1. Store - a unique Id for each store
2. DayOfWeek - literally day in a week, monday = 1, tuesday = 2,  etc.
3. Sales - the turnover for any given day (this is what you are predicting)
4. Customers - the number of customers on a given day
5. Open - an indicator for whether the store was open: 0 = closed, 1 = open
6. Promo - indicates whether a store is running a promo on that day
7. StateHoliday - indicates a state holiday. Normally all stores, with few exceptions, are closed on state holidays. Note that all schools are closed on public holidays and weekends. a = public holiday, b = Easter holiday, c = Christmas, 0 = None
8. SchoolHoliday - indicates if the (Store, Date) was affected by the closure of public schools

Ok let's randomly pick one of the row, 2nd index item, we are looking at store 3, it's on 2015 July 31st, there were 821 customers on that day, and the store has made 8314 Euros, I am guessing euros since its a germany drug store, store was opened even though there was a school holiday, there was also a promo on that day.

What else do we want to know? Maybe we want to know how big is our train set and also the time period for these sales.


```python
# how big is train set?
train.shape
```




    (1017209, 9)




```python
# what was the time period of the sales data?
print('from', train.Date.min(), 'to', train.Date.max())
```

    from 2013-01-01 to 2015-07-31


Alright I think we have sorted what we want to know about the `train` table, let's look at the next table `store`.


```python
store.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>a</td>
      <td>a</td>
      <td>14130.0</td>
      <td>12.0</td>
      <td>2006.0</td>
      <td>1</td>
      <td>14.0</td>
      <td>2011.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
  </tbody>
</table>
</div>



1. StoreType - differentiates between 4 different store models: a, b, c, d
2. Assortment - describes an assortment level: a = basic, b = extra, c = extended
3. CompetitionDistance - distance in meters to the nearest competitor store
4. CompetitionOpenSince[Month/Year] - gives the approximate year and month of the time the nearest competitor was opened
5. Promo - indicates whether a store is running a promo on that day
6. Promo2 - Promo2 is a continuing and consecutive promotion for some stores: 0 = store is not participating, 1 = store is participating
7. Promo2Since[Year/Week] - describes the year and calendar week when the store started participating in Promo2
8. PromoInterval - describes the consecutive intervals Promo2 is started, naming the months the promotion is started anew. E.g. "Feb,May,Aug,Nov" means each round starts in February, May, August, November of any given year for that store

Apparently `Store` table gives us information about the store, competition, and promo.


```python
store.shape
```




    (1115, 10)



There are 1,115 Rossmann store, so there's 1,115 rows of store information for each store.


```python
test.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>7</td>
      <td>4</td>
      <td>2015-09-17</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



All the columns in the `test` set have been discussed except `Id`, Id column represents a (Store, Date) duple within the test set. As expected, there's no `Sales` or `Customer` column, because these are the variables we would want to predict.

### External Dataset

These tables are provided by other Kagglers on the Rossmann competition, let's look at the first one `store_states`.


```python
store_states.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>HE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>TH</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NW</td>
    </tr>
  </tbody>
</table>
</div>



This basically tells us the which state the store is in, the state is in abbreviation, we need to get the full name of the state, which will be in the `state_names` table.


```python
store_states.shape
```




    (1115, 2)



As discussed, there were 1,115 stores, so there were 1,115 rows.


```python
state_names.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>StateName</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BadenWuerttemberg</td>
      <td>BW</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bayern</td>
      <td>BY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berlin</td>
      <td>BE</td>
    </tr>
  </tbody>
</table>
</div>




```python
state_names.shape
```




    (16, 2)



so we have got 16 state names, 1 for each of the state in the Germany.

Let's look at the next table `googletrend`.


```python
googletrend.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>week</th>
      <th>trend</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-02 - 2012-12-08</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-09 - 2012-12-15</td>
      <td>95</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-16 - 2012-12-22</td>
      <td>91</td>
    </tr>
  </tbody>
</table>
</div>



Not entirely sure how to make use of this table yet, but it seems to be giving information about some sort of Google trend, will talk about the naming convention of the `file` column later.



Lastly we've got `weather` table.


```python
weather.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>Date</th>
      <th>Max_TemperatureC</th>
      <th>Mean_TemperatureC</th>
      <th>Min_TemperatureC</th>
      <th>Dew_PointC</th>
      <th>MeanDew_PointC</th>
      <th>Min_DewpointC</th>
      <th>Max_Humidity</th>
      <th>Mean_Humidity</th>
      <th>...</th>
      <th>Max_VisibilityKm</th>
      <th>Mean_VisibilityKm</th>
      <th>Min_VisibilitykM</th>
      <th>Max_Wind_SpeedKm_h</th>
      <th>Mean_Wind_SpeedKm_h</th>
      <th>Max_Gust_SpeedKm_h</th>
      <th>Precipitationmm</th>
      <th>CloudCover</th>
      <th>Events</th>
      <th>WindDirDegrees</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-01</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>94</td>
      <td>87</td>
      <td>...</td>
      <td>31.0</td>
      <td>12.0</td>
      <td>4.0</td>
      <td>39</td>
      <td>26</td>
      <td>58.0</td>
      <td>5.08</td>
      <td>6.0</td>
      <td>Rain</td>
      <td>215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-02</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>93</td>
      <td>85</td>
      <td>...</td>
      <td>31.0</td>
      <td>14.0</td>
      <td>10.0</td>
      <td>24</td>
      <td>16</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>6.0</td>
      <td>Rain</td>
      <td>225</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-03</td>
      <td>11</td>
      <td>8</td>
      <td>6</td>
      <td>10</td>
      <td>8</td>
      <td>4</td>
      <td>100</td>
      <td>93</td>
      <td>...</td>
      <td>31.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>26</td>
      <td>21</td>
      <td>NaN</td>
      <td>1.02</td>
      <td>7.0</td>
      <td>Rain</td>
      <td>240</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
weather['file'].nunique()
```




    16



`weather` table gives us information about the daily weather information of each state, not going to pretend I am a weather expert and knows what each of these column means.

Alright, now that we have quickly glanced through the tables, let's prepare the data!

## Data preparation / Feature engineering

Firstly, we turn `StateHoliday` to booleans, to make them more convenient for modeling. We can do calculations on pandas fields using notation very similar (often identical) to numpy.


```python
train.StateHoliday = train.StateHoliday!='0'
test.StateHoliday = test.StateHoliday!='0'
```


```python
train.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>8314</td>
      <td>821</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the original dataset, `StateHoliday` was using various integers to represent different kind of State Holiday, but we are only interested in whether there was a holiday or not, so we turn it into booleans.

### Joins

`join_df` is a function for joining tables on specific fields. By default, we'll be doing a left outer join of `right` on the `left` argument using the given fields for each table.

Pandas does joins using the `merge` method. The `suffixes` argument describes the naming convention for duplicate fields. We've elected to leave the duplicate field names on the left untouched, and append a "\_y" to those on the right.

It's helpful to define `join_df` although we can straightaway use `pd.merge` because we are always going to use left join, and we want to avoid typing the same suffix everything we do join.


```python
def join_df(left, right, left_on, right_on=None, suffix='_y'):
    if right_on is None: right_on = left_on
    return left.merge(right, how='left', left_on=left_on, right_on=right_on,
                      suffixes=("", suffix))
```

Join weather/state names.


```python
# Equivalent to weather = weather.merge(state_names, how = 'left', left_on = "file", right_on = "StateName")
weather = join_df(weather, state_names, "file", "StateName")
```


```python
# Check new `weather` table
weather.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>Date</th>
      <th>Max_TemperatureC</th>
      <th>Mean_TemperatureC</th>
      <th>Min_TemperatureC</th>
      <th>Dew_PointC</th>
      <th>MeanDew_PointC</th>
      <th>Min_DewpointC</th>
      <th>Max_Humidity</th>
      <th>Mean_Humidity</th>
      <th>...</th>
      <th>Min_VisibilitykM</th>
      <th>Max_Wind_SpeedKm_h</th>
      <th>Mean_Wind_SpeedKm_h</th>
      <th>Max_Gust_SpeedKm_h</th>
      <th>Precipitationmm</th>
      <th>CloudCover</th>
      <th>Events</th>
      <th>WindDirDegrees</th>
      <th>StateName</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-01</td>
      <td>8</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
      <td>94</td>
      <td>87</td>
      <td>...</td>
      <td>4.0</td>
      <td>39</td>
      <td>26</td>
      <td>58.0</td>
      <td>5.08</td>
      <td>6.0</td>
      <td>Rain</td>
      <td>215</td>
      <td>NordrheinWestfalen</td>
      <td>NW</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-02</td>
      <td>7</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>93</td>
      <td>85</td>
      <td>...</td>
      <td>10.0</td>
      <td>24</td>
      <td>16</td>
      <td>NaN</td>
      <td>0.00</td>
      <td>6.0</td>
      <td>Rain</td>
      <td>225</td>
      <td>NordrheinWestfalen</td>
      <td>NW</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NordrheinWestfalen</td>
      <td>2013-01-03</td>
      <td>11</td>
      <td>8</td>
      <td>6</td>
      <td>10</td>
      <td>8</td>
      <td>4</td>
      <td>100</td>
      <td>93</td>
      <td>...</td>
      <td>2.0</td>
      <td>26</td>
      <td>21</td>
      <td>NaN</td>
      <td>1.02</td>
      <td>7.0</td>
      <td>Rain</td>
      <td>240</td>
      <td>NordrheinWestfalen</td>
      <td>NW</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 26 columns</p>
</div>



For the `googletrend` table, we would like to split out the `week` which is a date interval into a date object so that we can join it with other tables. Inside a series object we can access its `.str` attribute that gives access to all the string processing functions.

we can add new columns to a dataframe by simply defining it. We'll do this for googletrends by extracting dates and state names from the given data and adding those columns.

### `.Str.split`


```python
# Split the week into two columns, expand = True makes it a dataframe, [0] takes first column of the dataframe
googletrend['Date'] = googletrend.week.str.split(' - ', expand=True)[0]

# Split 'file' using string processing, get the 3rd column which is the state names
googletrend['State'] = googletrend.file.str.split('_', expand=True)[2]
```


```python
set(googletrend.State)
```




    {'BE',
     'BW',
     'BY',
     'HE',
     'HH',
     'NI',
     'NW',
     None,
     'RP',
     'SH',
     'SL',
     'SN',
     'ST',
     'TH'}




```python
set(state_names.State)
```




    {'BB',
     'BE',
     'BW',
     'BY',
     'HB',
     'HB,NI',
     'HE',
     'HH',
     'MV',
     'NW',
     'RP',
     'SH',
     'SL',
     'SN',
     'ST',
     'TH'}



It seems like the `googletrend` table does not have the `googletrend` for all the states, and it also has `None` which we can assume to be missing data.

For consistency purpose, we're also going to replace all instances of state name 'NI' to match the usage in the rest of the data: 'HB,NI'. This is a good opportunity to highlight pandas indexing. We can use `.loc[rows, cols]` to select a list of rows and a list of columns from the dataframe. In this case, we're selecting rows w/ statename 'NI' by using a boolean list `googletrend.State=='NI'` and selecting "State".


```python
# Set 'HB, NI' for rows with state name 'NI'
googletrend.loc[googletrend.State=='NI', "State"] = 'HB,NI'
```


```python
# Sanity check
googletrend.loc[googletrend.State =='HB,NI'].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>week</th>
      <th>trend</th>
      <th>Date</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1184</th>
      <td>Rossmann_DE_NI</td>
      <td>2012-12-02 - 2012-12-08</td>
      <td>76</td>
      <td>2012-12-02</td>
      <td>HB,NI</td>
    </tr>
    <tr>
      <th>1185</th>
      <td>Rossmann_DE_NI</td>
      <td>2012-12-09 - 2012-12-15</td>
      <td>73</td>
      <td>2012-12-09</td>
      <td>HB,NI</td>
    </tr>
    <tr>
      <th>1186</th>
      <td>Rossmann_DE_NI</td>
      <td>2012-12-16 - 2012-12-22</td>
      <td>84</td>
      <td>2012-12-16</td>
      <td>HB,NI</td>
    </tr>
  </tbody>
</table>
</div>



### `add_datepart`

The following extracts particular date fields from a complete datetime for the purpose of constructing categoricals.

We should *always* consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, we can't capture any trend/cyclical behavior as a function of time at any of these granularities. We'll add to every table with a date field.


```python
def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    if time: attr = attr + ['Hour', 'Minute', 'Second']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    if drop: df.drop(fldname, axis=1, inplace=True)
```


```python
add_datepart(weather, "Date", drop=False)
add_datepart(googletrend, "Date", drop=False)
add_datepart(train, "Date", drop=False)
add_datepart(test, "Date", drop=False)
```

Let's look at `googletrend` to compare before and after `add_datapart`:


```python
googletrend.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>week</th>
      <th>trend</th>
      <th>Date</th>
      <th>State</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-02 - 2012-12-08</td>
      <td>96</td>
      <td>2012-12-02</td>
      <td>SN</td>
      <td>2012</td>
      <td>12</td>
      <td>48</td>
      <td>2</td>
      <td>6</td>
      <td>337</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1354406400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-09 - 2012-12-15</td>
      <td>95</td>
      <td>2012-12-09</td>
      <td>SN</td>
      <td>2012</td>
      <td>12</td>
      <td>49</td>
      <td>9</td>
      <td>6</td>
      <td>344</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1355011200</td>
    </tr>
  </tbody>
</table>
</div>



Previously, we only have the first 5 columns, but after we used `add_datapart` we now have many more columns on the different granularity of date time object.

**How is this useful?**

Think about it this way, purchasing behaviour changes on every months payday, supposed the payday is on the 15th of the month, it will be easier for say a neural net to pick up the spike on 15th of the day when we have a column `Day` which is equal to 15, than letting the neural net to get the information from `Date` column which consist of day, month and year.


```python
# check the unique items in `file` column
set(googletrend.file)
```




    {'Rossmann_DE',
     'Rossmann_DE_BE',
     'Rossmann_DE_BW',
     'Rossmann_DE_BY',
     'Rossmann_DE_HE',
     'Rossmann_DE_HH',
     'Rossmann_DE_NI',
     'Rossmann_DE_NW',
     'Rossmann_DE_RP',
     'Rossmann_DE_SH',
     'Rossmann_DE_SL',
     'Rossmann_DE_SN',
     'Rossmann_DE_ST',
     'Rossmann_DE_TH'}



The Google trends data has a special category for the whole of the Germany - we'll pull that out so we can use it explicitly.


```python
trend_de = googletrend[googletrend.file == 'Rossmann_DE']
```

So now, we have the `googletrend` by state, and also for the whole of the Germany.

Now we can outer join all of our data into a single dataframe. Recall that in outer joins everytime a value in the joining field on the left table does not have a corresponding value on the right table, the corresponding row in the new table has Null values for all right table fields. One way to check that all records are consistent and complete is to check for Null values post-join, as we do here.

**Aside: Why not just do an inner join?**

If we are assuming that all records are complete and match on the field we desire, an inner join will do the same thing as an outer join. However, in the event we are wrong or a mistake is made, an outer join followed by a null-check will catch it. (Comparing before/after # of rows for inner join is equivalent, but requires keeping track of before/after row #'s. Outer join is easier.)

### More Joins

The first join that we want to do is to join the `store` table with `store_states` so the location information of stores and in the `store` table.


```python
store.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
    </tr>
  </tbody>
</table>
</div>




```python
store_states.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>HE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>TH</td>
    </tr>
  </tbody>
</table>
</div>




```python
store = join_df(store, store_states, "Store")
len(store[store.State.isnull()])
```




    0




```python
store.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
      <td>TH</td>
    </tr>
  </tbody>
</table>
</div>



We have left joined `store` and `store_states` as `store`, there is also no null values in the `State` column.

Next, we want to join our `train` table with our new `store` table, so that all information is in the same table, and we would like to do the same thing with the `test` set too.


```python
train.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Year</th>
      <th>...</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>31</td>
      <td>4</td>
      <td>212</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>31</td>
      <td>4</td>
      <td>212</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1438300800</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 22 columns</p>
</div>




```python
joined = join_df(train, store, "Store")
joined_test = join_df(test, store, "Store")
len(joined[joined.StoreType.isnull()]),len(joined_test[joined_test.StoreType.isnull()])
```




    (0, 0)




```python
# check new table `joined`
joined.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Year</th>
      <th>...</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
      <td>TH</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 32 columns</p>
</div>



Quick recap, we have the following list of tables

- `train`
- `store`
- `store_states`
- `state_names`
- `googletrend`
- `weather`
- `test`

and we have done the joins below:

- `weather = weather + state_state_names`
- `store = store + store_states`
- `joined = train + store`
- `joined_test = test + store`

Left join the google trend as well on ["State", "Year", "Week"], we can't really use only 1 column to join the two dataframes, because `googletrend` is a weekly data, and `train` is a daily data. That's why we are using these 3 columns to join the two tables.


```python
joined.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Store</th>
      <th>DayOfWeek</th>
      <th>Date</th>
      <th>Sales</th>
      <th>Customers</th>
      <th>Open</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>Year</th>
      <th>...</th>
      <th>StoreType</th>
      <th>Assortment</th>
      <th>CompetitionDistance</th>
      <th>CompetitionOpenSinceMonth</th>
      <th>CompetitionOpenSinceYear</th>
      <th>Promo2</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2SinceYear</th>
      <th>PromoInterval</th>
      <th>State</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>5263</td>
      <td>555</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>c</td>
      <td>a</td>
      <td>1270.0</td>
      <td>9.0</td>
      <td>2008.0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>HE</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>6064</td>
      <td>625</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>a</td>
      <td>a</td>
      <td>570.0</td>
      <td>11.0</td>
      <td>2007.0</td>
      <td>1</td>
      <td>13.0</td>
      <td>2010.0</td>
      <td>Jan,Apr,Jul,Oct</td>
      <td>TH</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 32 columns</p>
</div>




```python
googletrend.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file</th>
      <th>week</th>
      <th>trend</th>
      <th>Date</th>
      <th>State</th>
      <th>Year</th>
      <th>Month</th>
      <th>Week</th>
      <th>Day</th>
      <th>Dayofweek</th>
      <th>Dayofyear</th>
      <th>Is_month_end</th>
      <th>Is_month_start</th>
      <th>Is_quarter_end</th>
      <th>Is_quarter_start</th>
      <th>Is_year_end</th>
      <th>Is_year_start</th>
      <th>Elapsed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-02 - 2012-12-08</td>
      <td>96</td>
      <td>2012-12-02</td>
      <td>SN</td>
      <td>2012</td>
      <td>12</td>
      <td>48</td>
      <td>2</td>
      <td>6</td>
      <td>337</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1354406400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Rossmann_DE_SN</td>
      <td>2012-12-09 - 2012-12-15</td>
      <td>95</td>
      <td>2012-12-09</td>
      <td>SN</td>
      <td>2012</td>
      <td>12</td>
      <td>49</td>
      <td>9</td>
      <td>6</td>
      <td>344</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>1355011200</td>
    </tr>
  </tbody>
</table>
</div>




```python
joined = join_df(joined, googletrend, ["State","Year", "Week"])
joined_test = join_df(joined_test, googletrend, ["State","Year", "Week"])
len(joined[joined.trend.isnull()]),len(joined_test[joined_test.trend.isnull()])
```




    (0, 0)



Now we have incorporated the `googletrend` by state in the table, but we have left out the trend for the whole of the Germany, let's do that now, we want to use a different `suffix` for the duplicated columns for the whole of the Germany (e.g. `'_DE'`), so we will use the built-in pandas `merge` function.


```python
joined = joined.merge(trend_de, how ='left', on = ["Year", "Week"], suffixes=('', '_DE'))
joined_test = joined_test.merge(trend_de, how = 'left', on = ["Year", "Week"], suffixes=('', '_DE'))
len(joined[joined.trend_DE.isnull()]),len(joined_test[joined_test.trend_DE.isnull()])
```




    (0, 0)




```python
joined.columns
```




    Index(['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
           'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Week', 'Day',
           'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
           'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
           'Elapsed', 'StoreType', 'Assortment', 'CompetitionDistance',
           'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State', 'file',
           'week', 'trend', 'Date_y', 'Month_y', 'Day_y', 'Dayofweek_y',
           'Dayofyear_y', 'Is_month_end_y', 'Is_month_start_y', 'Is_quarter_end_y',
           'Is_quarter_start_y', 'Is_year_end_y', 'Is_year_start_y', 'Elapsed_y',
           'file_DE', 'week_DE', 'trend_DE', 'Date_DE', 'State_DE', 'Month_DE',
           'Day_DE', 'Dayofweek_DE', 'Dayofyear_DE', 'Is_month_end_DE',
           'Is_month_start_DE', 'Is_quarter_end_DE', 'Is_quarter_start_DE',
           'Is_year_end_DE', 'Is_year_start_DE', 'Elapsed_DE'],
          dtype='object')



Now, let's join with the `weather` table, we can `join_df` because we do not want to use a different suffix.


```python
joined = join_df(joined, weather, ["State","Date"])
joined_test = join_df(joined_test, weather, ["State","Date"])
len(joined[joined.Mean_TemperatureC.isnull()]),len(joined_test[joined_test.Mean_TemperatureC.isnull()])
```




    (0, 0)



Next we are going to remove some duplicated columns, we are left with a lot of them after so many joins we did. Let's have a look at all the columns we have now in the `joined` table.


```python
joined.columns
```




    Index(['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
           'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Week', 'Day',
           'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
           'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
           'Elapsed', 'StoreType', 'Assortment', 'CompetitionDistance',
           'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State', 'file',
           'week', 'trend', 'Date_y', 'Month_y', 'Day_y', 'Dayofweek_y',
           'Dayofyear_y', 'Is_month_end_y', 'Is_month_start_y', 'Is_quarter_end_y',
           'Is_quarter_start_y', 'Is_year_end_y', 'Is_year_start_y', 'Elapsed_y',
           'file_DE', 'week_DE', 'trend_DE', 'Date_DE', 'State_DE', 'Month_DE',
           'Day_DE', 'Dayofweek_DE', 'Dayofyear_DE', 'Is_month_end_DE',
           'Is_month_start_DE', 'Is_quarter_end_DE', 'Is_quarter_start_DE',
           'Is_year_end_DE', 'Is_year_start_DE', 'Elapsed_DE', 'file_y',
           'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC',
           'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC', 'Max_Humidity',
           'Mean_Humidity', 'Min_Humidity', 'Max_Sea_Level_PressurehPa',
           'Mean_Sea_Level_PressurehPa', 'Min_Sea_Level_PressurehPa',
           'Max_VisibilityKm', 'Mean_VisibilityKm', 'Min_VisibilitykM',
           'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'Max_Gust_SpeedKm_h',
           'Precipitationmm', 'CloudCover', 'Events', 'WindDirDegrees',
           'StateName', 'Year_y', 'Month_y', 'Week_y', 'Day_y', 'Dayofweek_y',
           'Dayofyear_y', 'Is_month_end_y', 'Is_month_start_y', 'Is_quarter_end_y',
           'Is_quarter_start_y', 'Is_year_end_y', 'Is_year_start_y', 'Elapsed_y'],
          dtype='object')



I will just randomly pick a few columns to understand more before we drop some of the duplicated columns.


```python
joined[['Date', 'Date_y', 'Day', 'Day_y', 'Month', 'Month_y', 'Elapsed', 'Elapsed_y']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Date_y</th>
      <th>Day</th>
      <th>Day_y</th>
      <th>Day_y</th>
      <th>Month</th>
      <th>Month_y</th>
      <th>Month_y</th>
      <th>Elapsed</th>
      <th>Elapsed_y</th>
      <th>Elapsed_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2015-07-31</td>
      <td>2015-08-02</td>
      <td>31</td>
      <td>2</td>
      <td>31</td>
      <td>7</td>
      <td>8</td>
      <td>7</td>
      <td>1438300800</td>
      <td>1438473600</td>
      <td>1438300800</td>
    </tr>
  </tbody>
</table>
</div>



Some of the columns have duplicated names, but not duplicated values, why? Because we merged weekly data with daily data. For instance, if we look at the first row, we have different values for `Date` and `Date_y` because `Date_y` consist of a weekly value that applies to the week `2015-07-27` to `2015-08-02`.

Now we are going to clean up the columns by removing all the duplicated columns (i.e. the ones with `'_y'`).


```python
# removing all the columns that ends with _y in `joined` and `joined_test`
for df in (joined, joined_test):
    for c in df.columns:
        if c.endswith('_y'):
            if c in df.columns: df.drop(c, inplace=True, axis=1)
```


```python
# Sanity check
joined.columns
```




    Index(['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
           'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Week', 'Day',
           'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
           'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
           'Elapsed', 'StoreType', 'Assortment', 'CompetitionDistance',
           'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State', 'file',
           'week', 'trend', 'file_DE', 'week_DE', 'trend_DE', 'Date_DE',
           'State_DE', 'Month_DE', 'Day_DE', 'Dayofweek_DE', 'Dayofyear_DE',
           'Is_month_end_DE', 'Is_month_start_DE', 'Is_quarter_end_DE',
           'Is_quarter_start_DE', 'Is_year_end_DE', 'Is_year_start_DE',
           'Elapsed_DE', 'Max_TemperatureC', 'Mean_TemperatureC',
           'Min_TemperatureC', 'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC',
           'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
           'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',
           'Min_Sea_Level_PressurehPa', 'Max_VisibilityKm', 'Mean_VisibilityKm',
           'Min_VisibilitykM', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
           'Max_Gust_SpeedKm_h', 'Precipitationmm', 'CloudCover', 'Events',
           'WindDirDegrees', 'StateName'],
          dtype='object')



Looking good, no more columns that ends with `'_y'`.

Next we'll fill in missing values to avoid complications with `NA`'s. `NA` (not available) is how Pandas indicates missing values; many models have problems when missing values are present, so it's always important to think about how to deal with them. In these cases, we are picking an arbitrary *signal value* that doesn't otherwise appear in the data.


```python
for df in (joined,joined_test):
    df['CompetitionOpenSinceYear'] = df.CompetitionOpenSinceYear.fillna(1900).astype(np.int32)
    df['CompetitionOpenSinceMonth'] = df.CompetitionOpenSinceMonth.fillna(1).astype(np.int32)
    df['Promo2SinceYear'] = df.Promo2SinceYear.fillna(1900).astype(np.int32)
    df['Promo2SinceWeek'] = df.Promo2SinceWeek.fillna(1).astype(np.int32)
```

Next we'll extract features "CompetitionOpenSince" and "CompetitionDaysOpen". Note the use of `apply()` in mapping a function across dataframe values.


```python
for df in (joined,joined_test):
    df["CompetitionOpenSince"] = pd.to_datetime(dict(year=df.CompetitionOpenSinceYear,
                                                     month=df.CompetitionOpenSinceMonth, day=15))
    df["CompetitionDaysOpen"] = df.Date.subtract(df.CompetitionOpenSince).dt.days
```

We'll replace some erroneous / outlying data.


```python
for df in (joined,joined_test):
    df.loc[df.CompetitionDaysOpen<0, "CompetitionDaysOpen"] = 0
    df.loc[df.CompetitionOpenSinceYear<1990, "CompetitionDaysOpen"] = 0
```

We add "CompetitionMonthsOpen" field, limiting the maximum to 2 years to limit number of unique categories. The reason for that is because this will be one of the categorical input variables in our neural net, and we do not want to put more categories than we should, so thats why we truncated it to limit it up to 2 years max.


```python
for df in (joined,joined_test):
    df["CompetitionMonthsOpen"] = df["CompetitionDaysOpen"]//30
    df.loc[df.CompetitionMonthsOpen>24, "CompetitionMonthsOpen"] = 24
joined.CompetitionMonthsOpen.unique()
```




    array([24,  3, 19,  9,  0, 16, 17,  7, 15, 22, 11, 13,  2, 23, 12,  4, 10,  1, 14, 20,  8, 18,  6, 21,  5])



We certainly can treat it as a continuous input variable, but we want to explicitly tell our neural net that a competition that is opened for 1 month and competition that is opened for 12 months are more different than it should be. Otherwise, it'd be hard for the neural net to find a functional form that capture the big difference between the competition of 1 month vs competition of 12 months.

Same process for Promo dates. You may need to install the `isoweek` package first.


```python
# If needed, uncomment:
# ! pip install isoweek
from isoweek import Week
```

With `Promo2SinceYear` and `Promo2SinceWeek`, we know which year and week, but in order to convert the date to a time stamp, we need `year, month, and day`, so we will use `Week` to help us obtain the data we need to convert.


```python
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
```

If you aren't cool enough to use `lambda`, you can do it this way, its the same:


```python
## Uncomment below to run code
# def create_promo2since(x):
#   return Week(x.Promo2SinceYear, x.Promo2SinceWeek).monday()

# for df in (joined, joined_test):
#     df["Promo2Since"] = pd.to_datetime(df.apply(create_promo2since, axis=1))
#     df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
```

Here's what the code above does:
`Week` basically takes in `Year`, `Weeks`, this tells which year we are looking at, and how many weeks we are into the year.
`.monday()` By adding this to `Week`, we are saying that we want to take every Monday, of given year, and the given weeks in the year.

So for example:


```python
joined[['Date', 'Promo2SinceYear', 'Promo2SinceWeek', 'Promo2Since', 'Promo2Days']].head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Date</th>
      <th>Promo2SinceYear</th>
      <th>Promo2SinceWeek</th>
      <th>Promo2Since</th>
      <th>Promo2Days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-07-31</td>
      <td>1900</td>
      <td>1</td>
      <td>1900-01-01</td>
      <td>42214</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-07-31</td>
      <td>2010</td>
      <td>13</td>
      <td>2010-03-29</td>
      <td>1950</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-07-31</td>
      <td>2011</td>
      <td>14</td>
      <td>2011-04-04</td>
      <td>1579</td>
    </tr>
  </tbody>
</table>
</div>



If we look at the 2nd row (index 1), the `Year` is 2010, and `Week` is 13, and we specified `.monday()`, after we get all that, we convert it to a timestamp with `pd.to_datetime`. This tells Python that we want to get timestamp of the 13th week monday on 2010, which gives us `2010-03-29`.

`Promo2Days` gives us the number of days it has been since promo2 has first started until date of sales.

Last piece of cleaning for the engineered features.


```python
joined.columns
```




    Index(['Store', 'DayOfWeek', 'Date', 'Sales', 'Customers', 'Open', 'Promo',
           'StateHoliday', 'SchoolHoliday', 'Year', 'Month', 'Week', 'Day',
           'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
           'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start',
           'Elapsed', 'StoreType', 'Assortment', 'CompetitionDistance',
           'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 'Promo2',
           'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval', 'State', 'file',
           'week', 'trend', 'file_DE', 'week_DE', 'trend_DE', 'Date_DE',
           'State_DE', 'Month_DE', 'Day_DE', 'Dayofweek_DE', 'Dayofyear_DE',
           'Is_month_end_DE', 'Is_month_start_DE', 'Is_quarter_end_DE',
           'Is_quarter_start_DE', 'Is_year_end_DE', 'Is_year_start_DE',
           'Elapsed_DE', 'Max_TemperatureC', 'Mean_TemperatureC',
           'Min_TemperatureC', 'Dew_PointC', 'MeanDew_PointC', 'Min_DewpointC',
           'Max_Humidity', 'Mean_Humidity', 'Min_Humidity',
           'Max_Sea_Level_PressurehPa', 'Mean_Sea_Level_PressurehPa',
           'Min_Sea_Level_PressurehPa', 'Max_VisibilityKm', 'Mean_VisibilityKm',
           'Min_VisibilitykM', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h',
           'Max_Gust_SpeedKm_h', 'Precipitationmm', 'CloudCover', 'Events',
           'WindDirDegrees', 'StateName', 'CompetitionOpenSince',
           'CompetitionDaysOpen', 'CompetitionMonthsOpen', 'Promo2Since',
           'Promo2Days'],
          dtype='object')




```python
for df in (joined,joined_test):
    df.loc[df.Promo2Days<0, "Promo2Days"] = 0 # this should strictly be positive because you cant run a future promo
    df.loc[df.Promo2SinceYear<1990, "Promo2Days"] = 0 # these are the missing values, we will just put them all to 0
    df["Promo2Weeks"] = df["Promo2Days"]//7 # new engineered feature, weeks since promo2
    df.loc[df.Promo2Weeks<0, "Promo2Weeks"] = 0  # stricly positive value
    df.loc[df.Promo2Weeks>25, "Promo2Weeks"] = 25 # truncate at 25 weeks, why? same reason as 'CompetitionMonthsOpen' previously
```

And we are done with the feature engineering and data cleaning! Let's convert all our progress to a pickle file so that we can import it easily in the future.


```python
joined.to_pickle(PATH/'joined')
joined_test.to_pickle(PATH/'joined_test')
```

If you are using Google Colab as your server and using the same setup as I did at the beginning of the blogpost, you should be able to download the pickle file like this

![pickle](https://i.imgur.com/uOzyu4w.png)

That is all for this blogpost, in the next one we will build a neural net on this dataset.
