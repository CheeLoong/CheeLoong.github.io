---
title: "Data Cleaning & Feature Enginnering on Rossmann"
date: 2019-05-13
permalink: /rossmann-clean2/
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

**Note: we are using `df.apply` which runs piece of python code over every row and its very slow, why do we still use it?**

In practice, whenever we try to apply a function to every row of something, or every element of a tensor, if there isn't a vectorized version function, we will have to call something like `df.apply` and pass in a lambda function like below.

In this case, we can't find a vectorized version function from pandas or numpy to convert Promo2SinceYear and Promo2SinceWeek into a date, so that's why we use the lambda approach.


```python
for df in (joined,joined_test):
    df["Promo2Since"] = pd.to_datetime(df.apply(lambda x: Week(
        x.Promo2SinceYear, x.Promo2SinceWeek).monday(), axis=1))
    df["Promo2Days"] = df.Date.subtract(df["Promo2Since"]).dt.days
```

If we are bad at programming and we need constant reminder on what `lambda` does, here's what it does behind the scene..


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

Let's call this a checkpoint by saving and converting all our progress to a pickle file so that we can import it easily in the future.


```python
joined.to_pickle(PATH/'joined')
joined_test.to_pickle(PATH/'joined_test')
```

If you are using Google Colab as your server and using the same setup as I did at the beginning of the blogpost, you should be able to download the pickle file like this

![pickle](https://i.imgur.com/uOzyu4w.png)

## Durations

Time series usually have events, and when a certain event happens, something interesting will happen before and after the event, think of grocery sales, if there's a holiday coming up, the grocery sales will most likely go up before the holiday (consumer buying supplies for holiday), and after the holiday (stock up groceries), and grocery sales most likely go down during the holiday.

Although it is not compulsory to do this sort of feature engineering when using neural net, is is often helpful to do so especially when we have limited data, or limited computation power. (*think of it as pointing the right direction to the neural net*)

Therefore when we have events associated with a time series data, it's often a good idea to create two new columns for each event:
1. How long is it going to be until the next event happens (*e.g. How long till the next state holiday*)
2. How long has it been since the last event happened (*e.g. How long has it been since the last state holiday*)


### `get_elapsed`

Essentially, when working with time series data to extract data that explains relationships across rows as opposed to columns, e.g.:
* Running averages
* Time until next event
* Time since last event

This is often difficult to do with most table manipulation frameworks, since they are designed to work with relationships across columns. As such, we've created a class to handle this type of data.

We'll define a function `get_elapsed` for cumulative counting across a sorted dataframe. Given a particular field `fld` to monitor, this function will start tracking time since the last occurrence of that field. When the field is seen again, the counter is set to zero.

Upon initialization, this will result in datetime na's until the field is encountered. This is reset every time a new store is seen. We'll see how to use this shortly.


```python
# initialize empty list to store results
def get_elapsed(fld, pre):
    day1 = np.timedelta64(1, 'D')
    last_date = np.datetime64()
    last_store = 0
    res = []
    for s,v,d in zip(df.Store.values,df[fld].values, df.Date.values):
        if s != last_store:
          last_date = np.datetime64()
          last_store = s  
        if v:
          last_date = d
        res.append(((d-last_date).astype('timedelta64[D]') / day1))  
    df[pre+fld] = res
```

Let's break this down line by line because we are *bad at programming and we LOVE to learn every single line of code we wrote*.


```
day1 = np.timedelta64(1, 'D')
```
This initializes a time delta of 1 day (think of it as 1 day time difference), why do we create this? Because we want make our result `res` to be without any unit measurement.



```
last_date = np.datetime64()
```
This initializes a Not a Time (NaT) value, which is equivalent of nan for timestamp values, we use it to reset `last_date` whenever we are iterating through a new store.

```
last_store = 0
```
We initialize this so that our first store (s = 1) will get `s != last_store` to run .


```
for s,v,d in zip(df.Store.values, df[fld].values, df.Date.values)
```

**`s`** is the store id, there should be a total of 1115 stores, so `s` takes on 1 to 1115.

**`v`** is stands for whatever your `fld` is, for example if your field is SchoolHoliday `fld = 'SchoolHoliday'`, `v` can be `1` meaning there was a school holiday or `0` whcih means no school holiday.

**`d`** is the date for the particular row information

*Note: s,v,d are arbitrary naming to get pieces of np arrays from zip, its fine to use other naming*

**What is `zip`?**

It's used to passed on iterables that we want to iterate through, in this case we want to iterate through the numpy arrays of each column in the dataframe.


**Why use `.values`?**

To turn each column of the dataframe (i.e. series) to numpy array. We do that because It's much faster to iterate through a numpy arrays than iterating a dataframe row (i.e. `for rows in df.iterrows()`)


```
if s != last_store:
    last_date = np.datetime64()
    last_store = s  
```

Whenever we have a new store, we reset `last_date` and update `last_store` to be the new store id.


```
if v:
    last_date = d
```
Whenever (fld e.g. SchoolHoliday) is 1, we update `last_date` to be the date of that iteration

```
res.append(((d-last_date).astype('timedelta64[D]') / day1))
```

If there's a holiday (`v == 1`) `last_date` will be updated and therefore we will always get `0.0`.

But if there isn't a holiday (`v == 0`) `last_date` will not be updated, and so `d - last_date` will give us the time difference between the iteration without holiday and the last time there was a holiday.


Now that we are smart and know what `get_elapsed` does, we'll be applying this to a subset of columns:


```python
columns = ["Date", "Store", "Promo", "StateHoliday", "SchoolHoliday"]
```


```python
# df = train[columns]
df = train[columns].append(test[columns])
```

Let's walk through an example.

Say we're looking at School Holiday. We'll first sort by Store, then Date, and then call `get_elapsed('SchoolHoliday', 'After')`:
This will apply to each row with School Holiday:
* A applied to every row of the dataframe in order of store and date
* Will add to the dataframe the days since seeing a School Holiday
* If we sort in the other direction, this will count the days until another holiday.


```python
fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
```


```python
get_elapsed(fld, 'After')
```


```python
fld = 'SchoolHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
```

Let's see if the function is working properly on our data.


```python
# check how the value change in the newly created columns
df[1058200:1058230]
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
      <th>Store</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>AfterSchoolHoliday</th>
      <th>BeforeSchoolHoliday</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>910169</th>
      <td>2013-04-07</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2.0</td>
      <td>-90.0</td>
    </tr>
    <tr>
      <th>911284</th>
      <td>2013-04-06</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>1.0</td>
      <td>-91.0</td>
    </tr>
    <tr>
      <th>912399</th>
      <td>2013-04-05</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>913514</th>
      <td>2013-04-04</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>914629</th>
      <td>2013-04-03</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>915744</th>
      <td>2013-04-02</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>916859</th>
      <td>2013-04-01</td>
      <td>1115</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>917974</th>
      <td>2013-03-31</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>919089</th>
      <td>2013-03-30</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>1.0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>920204</th>
      <td>2013-03-29</td>
      <td>1115</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>921319</th>
      <td>2013-03-28</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>922434</th>
      <td>2013-03-27</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>923549</th>
      <td>2013-03-26</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>924664</th>
      <td>2013-03-25</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>925779</th>
      <td>2013-03-24</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>72.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>926894</th>
      <td>2013-03-23</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>71.0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>928009</th>
      <td>2013-03-22</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>70.0</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>929124</th>
      <td>2013-03-21</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>69.0</td>
      <td>-4.0</td>
    </tr>
    <tr>
      <th>930239</th>
      <td>2013-03-20</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>68.0</td>
      <td>-5.0</td>
    </tr>
    <tr>
      <th>931354</th>
      <td>2013-03-19</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>67.0</td>
      <td>-6.0</td>
    </tr>
    <tr>
      <th>932469</th>
      <td>2013-03-18</td>
      <td>1115</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>66.0</td>
      <td>-7.0</td>
    </tr>
    <tr>
      <th>933584</th>
      <td>2013-03-17</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>65.0</td>
      <td>-8.0</td>
    </tr>
    <tr>
      <th>934699</th>
      <td>2013-03-16</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>64.0</td>
      <td>-9.0</td>
    </tr>
    <tr>
      <th>935814</th>
      <td>2013-03-15</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>63.0</td>
      <td>-10.0</td>
    </tr>
    <tr>
      <th>936929</th>
      <td>2013-03-14</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>62.0</td>
      <td>-11.0</td>
    </tr>
    <tr>
      <th>938044</th>
      <td>2013-03-13</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>61.0</td>
      <td>-12.0</td>
    </tr>
    <tr>
      <th>939159</th>
      <td>2013-03-12</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>60.0</td>
      <td>-13.0</td>
    </tr>
    <tr>
      <th>940274</th>
      <td>2013-03-11</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>59.0</td>
      <td>-14.0</td>
    </tr>
    <tr>
      <th>941389</th>
      <td>2013-03-10</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>58.0</td>
      <td>-15.0</td>
    </tr>
    <tr>
      <th>942504</th>
      <td>2013-03-09</td>
      <td>1115</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>57.0</td>
      <td>-16.0</td>
    </tr>
  </tbody>
</table>
</div>



We'll do this for two more fields.


```python
fld = 'StateHoliday'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
```


```python
fld = 'Promo'
df = df.sort_values(['Store', 'Date'])
get_elapsed(fld, 'After')
df = df.sort_values(['Store', 'Date'], ascending=[True, False])
get_elapsed(fld, 'Before')
```


```python
# Let's check our new columns
df.head(3)
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
      <th>Store</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>AfterSchoolHoliday</th>
      <th>BeforeSchoolHoliday</th>
      <th>AfterStateHoliday</th>
      <th>BeforeStateHoliday</th>
      <th>AfterPromo</th>
      <th>BeforePromo</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-09-17</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>13.0</td>
      <td>NaN</td>
      <td>105.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>856</th>
      <td>2015-09-16</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>12.0</td>
      <td>NaN</td>
      <td>104.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1712</th>
      <td>2015-09-15</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>11.0</td>
      <td>NaN</td>
      <td>103.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



We're going to set the active index to Date.


```python
df = df.set_index("Date")
```

Then set null values from elapsed field calculations to 0.


```python
columns = ['SchoolHoliday', 'StateHoliday', 'Promo']
```


```python
for o in ['Before', 'After']:
    for p in columns:
        a = o+p
        df[a] = df[a].fillna(0).astype(int)
```

### Windows / Rolling

Next we'll demonstrate window functions in pandas to calculate rolling quantities, where you apply some functions to some window interval of each datapoint.

Here we're sorting by date (`sort_index()`) and counting the number of events of interest (`sum()`) defined in `columns` in the following week (`rolling()`), grouped by Store (`groupby()`). We do the same in the opposite direction.


```python
columns
```




    ['SchoolHoliday', 'StateHoliday', 'Promo']




```python
# Use this output with `bwd` dataframe and `fwd` dataframe to understand .rolling()
df[['Store']+columns][df['Store'] == 1].sort_index().head(20)
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
      <th>SchoolHoliday</th>
      <th>StateHoliday</th>
      <th>Promo</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2013-01-01</th>
      <td>1</td>
      <td>1</td>
      <td>True</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-07</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-01-08</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-01-09</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-01-10</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-01-11</th>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2013-01-12</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-13</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-14</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-15</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-16</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-17</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-18</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-19</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2013-01-20</th>
      <td>1</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# bwd for backward rolling :)
bwd = df[['Store']+columns].sort_index().groupby("Store").rolling(7, min_periods=1).sum()
bwd.head(15)
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
      <th></th>
      <th>Store</th>
      <th>SchoolHoliday</th>
      <th>StateHoliday</th>
      <th>Promo</th>
    </tr>
    <tr>
      <th>Store</th>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="15" valign="top">1</th>
      <th>2013-01-01</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>4.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-07</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-08</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-09</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-10</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-11</th>
      <td>7.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-12</th>
      <td>7.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-13</th>
      <td>7.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-14</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-15</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



`.rolling(7, min_periods = 1)` means taking 7 days window and doing the aggregrate function that we specified (in this case `.sum()`), the `min_periods` basically applies to the edge of the data, which is the minimum acceptable window towards the end of the data for aggregrate function to apply.

For example:

If we look at *store 1*, on `2013-01-12`, the 7 days windows are `2013-01-06` until `2013-01-12`, In the 7 days windows, 5 days had `promo`, and that is why in the rolling output, `Promo` column on `2013-01-12` has a value of 5.

----
So that was 7 days windows towards the past, what about 7 days windows towards the future?

Let's look at the same example again:

If we look at *store 1*, on `2013-01-12`, the 7 days windows into the future are `2013-01-12` until `2013-01-18`, in the 7 days windows, none of the day had a promo. Thus, `Promo` column on `2013-01-12` has a value of 0.





```python
# fwd for forward rolling :)
fwd = df[['Store']+columns].sort_index(ascending=False).groupby("Store").rolling(7, min_periods=1).sum()
fwd.tail(20)
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
      <th></th>
      <th>Store</th>
      <th>SchoolHoliday</th>
      <th>StateHoliday</th>
      <th>Promo</th>
    </tr>
    <tr>
      <th>Store</th>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="20" valign="top">1115</th>
      <th>2013-01-20</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-19</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-18</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-17</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-16</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-15</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-14</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-13</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-12</th>
      <td>7805.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2013-01-11</th>
      <td>7805.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2013-01-10</th>
      <td>7805.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-09</th>
      <td>7805.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-08</th>
      <td>7805.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-07</th>
      <td>7805.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-06</th>
      <td>7805.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-05</th>
      <td>7805.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>2013-01-04</th>
      <td>7805.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2013-01-03</th>
      <td>7805.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2013-01-02</th>
      <td>7805.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2013-01-01</th>
      <td>7805.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



Next we want to drop the Store indices grouped together in the window function, because we are not interested in rolling function on the store id.

Often in pandas, there is an option to do this in place. This is time and memory efficient when working with large datasets. We will also reset to our default index.


```python
bwd.drop('Store',1,inplace=True)
bwd.reset_index(inplace=True)
```


```python
fwd.drop('Store',1,inplace=True)
fwd.reset_index(inplace=True)
```


```python
df.reset_index(inplace=True)
```

### One Last Joins

Now we'll merge these values onto the df.


```python
df = df.merge(bwd, 'left', ['Date', 'Store'], suffixes=['', '_bw'])
df = df.merge(fwd, 'left', ['Date', 'Store'], suffixes=['', '_fw'])
```


```python
df.columns
```




    Index(['Date', 'Store', 'Promo', 'StateHoliday', 'SchoolHoliday',
           'AfterSchoolHoliday', 'BeforeSchoolHoliday', 'AfterStateHoliday',
           'BeforeStateHoliday', 'AfterPromo', 'BeforePromo', 'SchoolHoliday_bw',
           'StateHoliday_bw', 'Promo_bw', 'SchoolHoliday_fw', 'StateHoliday_fw',
           'Promo_fw'],
          dtype='object')




```python
df.head(3)
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
      <th>Store</th>
      <th>Promo</th>
      <th>StateHoliday</th>
      <th>SchoolHoliday</th>
      <th>AfterSchoolHoliday</th>
      <th>BeforeSchoolHoliday</th>
      <th>AfterStateHoliday</th>
      <th>BeforeStateHoliday</th>
      <th>AfterPromo</th>
      <th>BeforePromo</th>
      <th>SchoolHoliday_bw</th>
      <th>StateHoliday_bw</th>
      <th>Promo_bw</th>
      <th>SchoolHoliday_fw</th>
      <th>StateHoliday_fw</th>
      <th>Promo_fw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-09-17</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>13</td>
      <td>0</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-09-16</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-09-15</td>
      <td>1</td>
      <td>1</td>
      <td>False</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>103</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.drop(columns,1,inplace=True)
```


```python
df.head()
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
      <th>Store</th>
      <th>AfterSchoolHoliday</th>
      <th>BeforeSchoolHoliday</th>
      <th>AfterStateHoliday</th>
      <th>BeforeStateHoliday</th>
      <th>AfterPromo</th>
      <th>BeforePromo</th>
      <th>SchoolHoliday_bw</th>
      <th>StateHoliday_bw</th>
      <th>Promo_bw</th>
      <th>SchoolHoliday_fw</th>
      <th>StateHoliday_fw</th>
      <th>Promo_fw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2015-09-17</td>
      <td>1</td>
      <td>13</td>
      <td>0</td>
      <td>105</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2015-09-16</td>
      <td>1</td>
      <td>12</td>
      <td>0</td>
      <td>104</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2015-09-15</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>103</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2015-09-14</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>102</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2015-09-13</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>101</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



It's usually a good idea to back up large tables of extracted / wrangled features before you join them onto another one, that way you can go back to it easily if you need to make changes to it.


```python
# after running this code, a pickle file should appear in `PATH`, download it if u wanna backup
df.to_pickle(PATH/'df')
```


```python
df["Date"] = pd.to_datetime(df.Date)
```


```python
df.columns
```




    Index(['Date', 'Store', 'AfterSchoolHoliday', 'BeforeSchoolHoliday',
           'AfterStateHoliday', 'BeforeStateHoliday', 'AfterPromo', 'BeforePromo',
           'SchoolHoliday_bw', 'StateHoliday_bw', 'Promo_bw', 'SchoolHoliday_fw',
           'StateHoliday_fw', 'Promo_fw'],
          dtype='object')




```python
joined = pd.read_pickle(PATH/'joined')
joined_test = pd.read_pickle(PATH/'joined_test')
```


```python
df.shape
```




    (1058297, 14)




```python
joined = join_df(joined, df, ['Store', 'Date'])
```


```python
joined_test = join_df(joined_test, df, ['Store', 'Date'])
```


```python
joined[joined.Sales == 0]
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
      <th>AfterStateHoliday</th>
      <th>BeforeStateHoliday</th>
      <th>AfterPromo</th>
      <th>BeforePromo</th>
      <th>SchoolHoliday_bw</th>
      <th>StateHoliday_bw</th>
      <th>Promo_bw</th>
      <th>SchoolHoliday_fw</th>
      <th>StateHoliday_fw</th>
      <th>Promo_fw</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>291</th>
      <td>292</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>875</th>
      <td>876</td>
      <td>5</td>
      <td>2015-07-31</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1406</th>
      <td>292</td>
      <td>4</td>
      <td>2015-07-30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>876</td>
      <td>4</td>
      <td>2015-07-30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>66</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2521</th>
      <td>292</td>
      <td>3</td>
      <td>2015-07-29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3105</th>
      <td>876</td>
      <td>3</td>
      <td>2015-07-29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>65</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>3636</th>
      <td>292</td>
      <td>2</td>
      <td>2015-07-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4220</th>
      <td>876</td>
      <td>2</td>
      <td>2015-07-28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>4751</th>
      <td>292</td>
      <td>1</td>
      <td>2015-07-27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5335</th>
      <td>876</td>
      <td>1</td>
      <td>2015-07-27</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>63</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5575</th>
      <td>1</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5576</th>
      <td>2</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5577</th>
      <td>3</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5578</th>
      <td>4</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5579</th>
      <td>5</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5580</th>
      <td>6</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5581</th>
      <td>7</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5582</th>
      <td>8</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5583</th>
      <td>9</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5584</th>
      <td>10</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>1</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5585</th>
      <td>11</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5586</th>
      <td>12</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5587</th>
      <td>13</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>-20</td>
      <td>9</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5588</th>
      <td>14</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5589</th>
      <td>15</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5590</th>
      <td>16</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5591</th>
      <td>17</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5592</th>
      <td>18</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>62</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5593</th>
      <td>19</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>0</td>
      <td>9</td>
      <td>-1</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>5594</th>
      <td>20</td>
      <td>7</td>
      <td>2015-07-26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>False</td>
      <td>0</td>
      <td>2015</td>
      <td>...</td>
      <td>52</td>
      <td>-20</td>
      <td>9</td>
      <td>-1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1017178</th>
      <td>1085</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017179</th>
      <td>1086</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017180</th>
      <td>1087</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017181</th>
      <td>1088</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017182</th>
      <td>1089</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017183</th>
      <td>1090</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017184</th>
      <td>1091</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017185</th>
      <td>1092</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017186</th>
      <td>1093</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017187</th>
      <td>1094</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017188</th>
      <td>1095</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017189</th>
      <td>1096</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017191</th>
      <td>1098</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017192</th>
      <td>1099</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017193</th>
      <td>1100</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017194</th>
      <td>1101</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017195</th>
      <td>1102</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017196</th>
      <td>1103</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017197</th>
      <td>1104</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017198</th>
      <td>1105</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017199</th>
      <td>1106</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017200</th>
      <td>1107</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017201</th>
      <td>1108</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017202</th>
      <td>1109</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017203</th>
      <td>1110</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017204</th>
      <td>1111</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017205</th>
      <td>1112</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017206</th>
      <td>1113</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017207</th>
      <td>1114</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1017208</th>
      <td>1115</td>
      <td>2</td>
      <td>2013-01-01</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
      <td>1</td>
      <td>2013</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-6</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>172871 rows × 92 columns</p>
</div>



If the idea is to predict sales, and the store would have zero sales when the store was closed, that can be considered as redundant data, we will omit this data from training.

However, that might not be the best idea, because if stores are closed for refurbishment, there are probably spikes in sales before and after these periods, by omitting the data frm training, we will give up the ability to leverage such information. (Maybe one day we will revisit this whole dataset without omitting the data).


```python
joined = joined[joined.Sales!=0]
```

We'll back this up as well.


```python
joined.reset_index(inplace=True)
joined_test.reset_index(inplace=True)
```


```python
joined.to_pickle(PATH/'train_clean')
joined_test.to_pickle(PATH/'test_clean')
```

Remember to download the pickle files from `PATH`!

That is all for this blogpost, in the next one we will be using the cleansed train and test set to build a neural net, thank you for reading the blogpost, and credit goes to [fast.ai](https://www.fast.ai/) for the materials.
