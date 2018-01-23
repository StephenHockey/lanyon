
For this project, we are using the Ames Housing Dataset that contains over 80 columns of different features relating to real estate sold in Ames, Iowa. This includes a sale price and whether a sale was abnormal or not, which we will use as targets for two different models.

# All Imports


```python
from sklearn.ensemble import AdaBoostClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, \
    cross_val_score, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelBinarizer, StandardScaler

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.linear_model import Ridge, Lasso, ElasticNet, \
    LinearRegression, LogisticRegression, RidgeCV, LassoCV, \
    ElasticNetCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV, \
    train_test_split
from sklearn.feature_selection import RFE, SelectKBest, f_classif, \
    f_regression
from sklearn.ensemble import GradientBoostingRegressor, \
    RandomForestRegressor, GradientBoostingClassifier, \
    RandomForestClassifier
    
from sklearn.metrics import confusion_matrix, classification_report, \
    roc_curve, auc
```


```python
training_data = \
    pd.read_csv('/Users/stephenhockey/Downloads/trainingdata.csv')
testing_data = pd.read_csv('/Users/stephenhockey/Downloads/test.csv')
```


```python
print(training_data.shape, testing_data.shape)
```

    (2051, 82) (879, 80)


training_data is the dataset downloaded from Kaggle and loaded into a dataframe that I will use to train my models. We are working with 2051 different real estate sales recorded with 82 different data points for each. At least, we hope we are! Let's check to see how many null values our data contains.


```python
nulls = training_data.isnull().sum().sort_values(ascending=False)
nulls = nulls.reset_index()
nulls.columns = ['Column', 'Nulls']
gtz = nulls['Nulls'] > 0
nulls = nulls[gtz]
nulls
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Column</th>
      <th>Nulls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pool QC</td>
      <td>2042</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Misc Feature</td>
      <td>1986</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Alley</td>
      <td>1911</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fence</td>
      <td>1651</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Fireplace Qu</td>
      <td>1000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Lot Frontage</td>
      <td>330</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Garage Cond</td>
      <td>114</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Garage Finish</td>
      <td>114</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Garage Yr Blt</td>
      <td>114</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Garage Qual</td>
      <td>114</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Garage Type</td>
      <td>113</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Bsmt Exposure</td>
      <td>58</td>
    </tr>
    <tr>
      <th>12</th>
      <td>BsmtFin Type 2</td>
      <td>56</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bsmt Cond</td>
      <td>55</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Bsmt Qual</td>
      <td>55</td>
    </tr>
    <tr>
      <th>15</th>
      <td>BsmtFin Type 1</td>
      <td>55</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Mas Vnr Type</td>
      <td>22</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Mas Vnr Area</td>
      <td>22</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Bsmt Half Bath</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Bsmt Full Bath</td>
      <td>2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Garage Cars</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Garage Area</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Total Bsmt SF</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Bsmt Unf SF</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>BsmtFin SF 2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>BsmtFin SF 1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Okay, so there is a decent amount of missing information, but it seems to be split up into categories (garage, basement, Mas Vnr) with very similar amounts missing between columns in each category. I'm going to split up dealing with the missing information by the category, starting with all of the columns with more than 300 missing values.

### Columns with Many Nulls


```python
# Make a list of columns with more than 300 null values
missing_cols = []
for col in training_data:
    if training_data[col].isnull().sum() > 300:
        missing_cols.append(col)
    else:
        missing_cols = missing_cols

missing_cols
```




    ['Lot Frontage', 'Alley', 'Fireplace Qu', 'Pool QC', 'Fence', 
    'Misc Feature']



**From the data dictionary that you can find here:** https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

LotFrontage: Linear feet of street connected to property

Alley: Type of alley access to property
* Grvl - Gravel
* Pave - Paved
* NA - No alley access

FireplaceQu: Fireplace quality
* Ex - Excellent
* Gd - Good
* TA - Average
* Fa - Fair
* Po - Poor
* NA - No Fireplace

PoolQC: Pool quality
* Ex - Excellent
* Gd - Good
* TA - Average/Typical
* Fa - Fair
* NA - No Pool

Fence: Fence quality
* GdPrv - Good Privacy
* MnPrv - Minimum Privacy
* GdWo - Good Wood
* MnWw - Minimum Wood/Wire
* NA - No Fence

MiscFeature: Miscellaneous feature not covered in other categories
* Elev - Elevator
* Gar2 - 2nd Garage (if not described in garage section)
* Othr - Other
* Shed - Shed (over 100 SF)
* TenC - Tennis Court
* NA - None

Of all of these columns, the only one where a null value does not mean 'does not have this' is Lot Frontage. Because of this, and the fact that lot frontage is just the number of feet of street touching the property, we will just drop Lot Frontage while keeping all of the others and changing their null values.


```python
training_data = training_data.drop('Lot Frontage', axis=1)

training_data['Alley'].fillna('No Alley', inplace=True)
training_data['Fireplace Qu'].fillna('No Fireplace', inplace=True)
training_data['Pool QC'].fillna('No Pool', inplace=True)
training_data['Fence'].fillna('No Fence', inplace=True)
training_data['Misc Feature'].fillna('No Misc Feature', inplace=True)
```

### Null Values in Garage Columns


```python
training_data.iloc[:, 58:65].isnull().sum()
```




    Garage Type      113
    Garage Yr Blt    114
    Garage Finish    114
    Garage Cars        1
    Garage Area        1
    Garage Qual      114
    Garage Cond      114
    dtype: int64



For the columns Garage Type, Garage Yr Built, Garage Finish, Garage Qual, and Garage Cond, the NaN values are also because that property does not have a garage. Therefore, we should see the same number of null values for all of these, but Garage Type has one less than the others. To confirm that there is a value for Garage Type when all of the others are null, we can filter the dataset looking for where Garage Type is not null and one of the others is null.


```python
training_data.iloc[:, 58:65][(training_data['Garage Type'].notnull()) & 
                             (training_data['Garage Qual'].isnull())]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Garage Type</th>
      <th>Garage Yr Blt</th>
      <th>Garage Finish</th>
      <th>Garage Cars</th>
      <th>Garage Area</th>
      <th>Garage Qual</th>
      <th>Garage Cond</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1712</th>
      <td>Detchd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



From that we can see that the Garage Type is categorized as Detached when all of the other garage variables are NaN.
This includes the only null values for both Garage Cars and Garage Area. I am assuming that the Garage Type is simply
misclassified (should be NaN for no garage) and because there is no garage both Garage Cars and Garage Area should be 0. We can get rid of all of these null values by creating a value of 'No Garage' for Garage Type, Garage Finish, Garage Qual, Garage Cond, and Garage Yr Blt to indicate that that house does not have a garage, instead of having a null value. Then we can change the two null values for Garage Cars and Garage Area to 0s, and the misclassified Garage Type (when Garage Type is not null but all of the others are null) to null before changing all null values in Garage Type, Garage Finish, Garage Qual, Garage Cond, and Garage Yr Blt to No Garage.


```python
training_data.loc[((training_data['Garage Type'].notnull()) & 
                   (training_data['Garage Qual'].isnull()) & 
                   (training_data['Garage Finish'].isnull()) & 
                   (training_data['Garage Yr Blt'].isnull()) &
                   (training_data['Garage Yr Blt'].isnull()), \
                   'Garage Type')] = np.nan 
```


```python
training_data.loc[(training_data['Garage Type'].isnull() & 
                   training_data['Garage Cars'].isnull()), \
                   'Garage Cars'] = 0

training_data.loc[(training_data['Garage Type'].isnull() & 
                   training_data['Garage Area'].isnull()), \
                   'Garage Area'] = 0
```


```python
training_data['Garage Type'].fillna('No Garage', inplace=True)
training_data['Garage Finish'].fillna('No Garage', inplace=True)
training_data['Garage Qual'].fillna('No Garage', inplace=True)
training_data['Garage Cond'].fillna('No Garage', inplace=True)
training_data['Garage Yr Blt'].fillna('No Garage', inplace=True)
```

### Null Values in Basement Columns


```python
training_data.iloc[:, 30:39].isnull().sum()
```




    Bsmt Qual         55
    Bsmt Cond         55
    Bsmt Exposure     58
    BsmtFin Type 1    55
    BsmtFin SF 1       1
    BsmtFin Type 2    56
    BsmtFin SF 2       1
    Bsmt Unf SF        1
    Total Bsmt SF      1
    dtype: int64




```python
training_data.iloc[:, 47:49].isnull().sum()
```




    Bsmt Full Bath    2
    Bsmt Half Bath    2
    dtype: int64



Again, for the columns Bsmt Qual, Bsmt Cond, Bsmt Exposure, BsmtFin Type 1, and BsmtFin Type 2, the null values signify the property not having a basement. While they should all match up, we see a few discrepancies in Bsmt Exposure and BsmtFin Type 2. We can dig a little more into those after we double check all of the other columns null values are occuring in the same rows.


```python
training_data.iloc[:, 29:36][(training_data['Bsmt Qual'].isnull()) & 
                    (training_data['Bsmt Exposure'].isnull()) &
                    (training_data['Bsmt Cond'].isnull()) & 
                    (training_data['BsmtFin Type 1'].isnull()) &
                    (training_data['BsmtFin Type 2'].isnull())].shape[0]
```




    55



From this we can see that there are 55 rows where all of the Basement columns with NaN values signifying that the property has no basement were null. This is exactly what we'd expect from looking at the null value counts above. However, there are a few cases in Bsmt Exposure and one case in BsmtFin Type 2 where there is a null value when Bsmt Qual and Cond are not null, which should not happen.


```python
# To look at where the Bsmt Exposure column is null when the 
# Bsmt Qual has a value, we can do the following:
training_data.iloc[:, 30:39][(training_data['Bsmt Qual'].notnull()) &
                             (training_data['Bsmt Exposure'].isnull())]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1456</th>
      <td>Gd</td>
      <td>TA</td>
      <td>NaN</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>725.0</td>
      <td>725.0</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>Gd</td>
      <td>TA</td>
      <td>NaN</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>1595.0</td>
      <td>1595.0</td>
    </tr>
    <tr>
      <th>1997</th>
      <td>Gd</td>
      <td>TA</td>
      <td>NaN</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>Unf</td>
      <td>0.0</td>
      <td>936.0</td>
      <td>936.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
training_data['Bsmt Exposure'].value_counts()
```




    No    1339
    Av     288
    Gd     203
    Mn     163
    Name: Bsmt Exposure, dtype: int64



As No Exposure is by far the highest occuring, it is a fair assumption to believe that the missing Bsmt Exposure values are also No Exposure, especially considering this dataset's tendency to have NaN values for 'none' values.


```python
training_data.loc[(training_data['Bsmt Qual'].notnull() & \
                   training_data['Bsmt Exposure'].isnull()), \
                   'Bsmt Exposure'] = 'No'

# Here we are using (training_data['Bsmt Qual'].notnull() & 
# training_data['Bsmt Exposure'].isnull()) as the row indexer
# to select all of the rows we want to change in the 'Bsmt Exposure' 
# column which is given as the column indexer, then
# setting all of those values to 'No' for no exposure.
```


```python
training_data.iloc[:, 30:39][(training_data['Bsmt Qual'].notnull()) & 
                             (training_data['BsmtFin Type 2'].isnull())]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1147</th>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>1124.0</td>
      <td>NaN</td>
      <td>479.0</td>
      <td>1603.0</td>
      <td>3206.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
(training_data['BsmtFin Type 1'] == training_data['BsmtFin Type 2']).mean()
```




    0.29449049244271086




```python
training_data['BsmtFin Type 1'].value_counts()
```




    GLQ    615
    Unf    603
    ALQ    293
    BLQ    200
    Rec    183
    LwQ    102
    Name: BsmtFin Type 1, dtype: int64




```python
training_data['BsmtFin Type 2'].value_counts()

# I would have assumed that most of the time the BsmtFin Type 1 and 
# BsmtFin Type 2 would be the same, but that is not the case with only 
# 29% of the rows having equal values for each. After looking at the 
# value counts in each column it becomes quite clear why, with almost 
# three times as many BsmtFin Type 2s being unfinished compared to the
# Type 1counterpart. Because of this it is a safe assumption to set the
# BsmtFin Type 2 value to Unf.
```




    Unf    1749
    Rec      80
    LwQ      60
    BLQ      48
    ALQ      35
    GLQ      23
    Name: BsmtFin Type 2, dtype: int64




```python
training_data.loc[(training_data['BsmtFin Type 1'].notnull() & 
                   training_data['BsmtFin Type 2'].isnull()), \
                   'BsmtFin Type 2'] = 'Unf'
```


```python
training_data.iloc[:, 30:39].isnull().sum()

# With all of the mismatched cases being dealt with, we can now just 
# change all of the null values in all of the columns with 55 null 
# values to a new classification 'No Basement'.
```




    Bsmt Qual         55
    Bsmt Cond         55
    Bsmt Exposure     55
    BsmtFin Type 1    55
    BsmtFin SF 1       1
    BsmtFin Type 2    55
    BsmtFin SF 2       1
    Bsmt Unf SF        1
    Total Bsmt SF      1
    dtype: int64




```python
training_data[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', \
        'BsmtFin Type 1', 'BsmtFin Type 2']] = \
training_data[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', \
        'BsmtFin Type 1', 'BsmtFin Type 2']].fillna('No Basement')
```


```python
# We can now check if those other 4 basement columns seen above with 
# one null value in each all occur within the same row.

training_data.iloc[:, 30:39][(training_data['BsmtFin SF 1'].isnull()) & 
                            (training_data['BsmtFin SF 2'].isnull()) &
                            (training_data['Bsmt Unf SF'].isnull()) & 
                            (training_data['Total Bsmt SF'].isnull())]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1327</th>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>NaN</td>
      <td>No Basement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Because they are all in the same row, on a property with no basement we can easily see that these are again a case of null values being assigned in place of putting a 0. These can be changed generally by doing the following:


```python
training_data.loc[((training_data['Bsmt Qual'] == 'No Basement') & 
            (training_data['Bsmt Cond'] == 'No Basement') &
            (training_data['Bsmt Exposure'] == 'No Basement') & 
            (training_data['BsmtFin Type 1'] == 'No Basement') & 
            (training_data['BsmtFin Type 2'] == 'No Basement'), \
            'BsmtFin SF 1')] = 0

training_data.loc[((training_data['Bsmt Qual'] == 'No Basement') & 
            (training_data['Bsmt Cond'] == 'No Basement') &
            (training_data['Bsmt Exposure'] == 'No Basement') & 
            (training_data['BsmtFin Type 1'] == 'No Basement') &
            (training_data['BsmtFin Type 2'] == 'No Basement'), \
            'BsmtFin SF 2')] = 0

training_data.loc[((training_data['Bsmt Qual'] == 'No Basement') & 
            (training_data['Bsmt Cond'] == 'No Basement') &
            (training_data['Bsmt Exposure'] == 'No Basement') & 
            (training_data['BsmtFin Type 1'] == 'No Basement') &
            (training_data['BsmtFin Type 2'] == 'No Basement'), \
            'Bsmt Unf SF')] = 0

training_data.loc[((training_data['Bsmt Qual'] == 'No Basement') & 
            (training_data['Bsmt Cond'] == 'No Basement') &
            (training_data['Bsmt Exposure'] == 'No Basement') & 
            (training_data['BsmtFin Type 1'] == 'No Basement') &
            (training_data['BsmtFin Type 2'] == 'No Basement'), \
            'Total Bsmt SF')] = 0
```

Finishing up with the null values in basement columns we have the two null values in each of Bsmt Half Bath and Bsmt Full Bath. Let's check to see if they are both in the same rows first of all, as it is probably another case of null values being inputted instead of a 0.


```python
training_data.iloc[:, 30:39][(training_data['Bsmt Full Bath'].isnull()) & 
                             (training_data['Bsmt Half Bath'].isnull())]

# From this we can see that those two columns are null when only in rows 
# in which the property has no basement. We can therefore impute values 
# of 0 into both missing Bsmt Full Bath and Bsmt Half Bath rows.
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Bsmt Qual</th>
      <th>Bsmt Cond</th>
      <th>Bsmt Exposure</th>
      <th>BsmtFin Type 1</th>
      <th>BsmtFin SF 1</th>
      <th>BsmtFin Type 2</th>
      <th>BsmtFin SF 2</th>
      <th>Bsmt Unf SF</th>
      <th>Total Bsmt SF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>616</th>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>0.0</td>
      <td>No Basement</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1327</th>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>No Basement</td>
      <td>0.0</td>
      <td>No Basement</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
training_data['Bsmt Half Bath'].fillna(value=0, inplace=True)
```


```python
training_data['Bsmt Full Bath'].fillna(value=0, inplace=True)
```

### Null Values in Masonry Veneer Columns


```python
training_data.iloc[:, 25:27].isnull().sum()
```




    Mas Vnr Type    22
    Mas Vnr Area    22
    dtype: int64




```python
# To double check that all 22 missing values in both Mas Vnr Type and 
# Mas Vnr Area occur in the same rows.

training_data.iloc[:, 24:26][(training_data['Mas Vnr Type'].isnull()) & 
                        (training_data['Mas Vnr Area'].isnull())].shape
```




    (22, 2)




```python
# As this dataset has a habit of mistakenly having null values in place 
# of 'None' values in applicable columns, I am going to assume that these 
# 22 properties do not have Masonry Veneers, and will thus change the 
# missing Mas Vnr Type cells to 'None' and the missing Mas Vnr Area cells
# to 0.

training_data['Mas Vnr Type'].fillna(value='None', inplace=True)
training_data['Mas Vnr Area'].fillna(value=0, inplace=True)
```


```python
training_data['Mas Vnr Type'].value_counts()
```




    None       1240
    BrkFace     630
    Stone       168
    BrkCmn       13
    Name: Mas Vnr Type, dtype: int64




```python
training_data['Mas Vnr Area'].value_counts().sort_index()[0]
```




    1238



Upon further inspection into the value counts of these two columns, we can see that the number of Mas Vnr Area equal to 0 does not match up with the number of Mas Vnr Type equal to None like it should. We can dive a little deeper into where these two sets do not equal eachother.


```python
training_data.iloc[:, 25:27][(training_data['Mas Vnr Type'] == 'None') & 
                             (training_data['Mas Vnr Area'] != 0)]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mas Vnr Type</th>
      <th>Mas Vnr Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>765</th>
      <td>None</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>810</th>
      <td>None</td>
      <td>288.0</td>
    </tr>
    <tr>
      <th>1148</th>
      <td>None</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1684</th>
      <td>None</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1832</th>
      <td>None</td>
      <td>344.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Three of these are obviously just mis-entered as 1.0 rather than 0 
# when there is no Masonry Veneer, which can be easily changed.

training_data.loc[(training_data['Mas Vnr Type'] == 'None') & 
            (training_data['Mas Vnr Area'] == 1), 'Mas Vnr Area'] = 0
```


```python
training_data['Mas Vnr Type'].value_counts()
```




    None       1240
    BrkFace     630
    Stone       168
    BrkCmn       13
    Name: Mas Vnr Type, dtype: int64




```python
# For the other 2 we will just set the None values where Mas Vnr Area 
# != 0 equal to the second most common classification, as None is by 
# far the most common.

training_data.loc[(training_data['Mas Vnr Type'] == 'None') & 
            (training_data['Mas Vnr Area'] != 0), 'Mas Vnr Type'] = \
            training_data['Mas Vnr Type'].value_counts().index[1]
```


```python
training_data.iloc[:, 25:27][(training_data['Mas Vnr Type'] != 'None') & 
                             (training_data['Mas Vnr Area'] == 0)]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mas Vnr Type</th>
      <th>Mas Vnr Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>669</th>
      <td>BrkFace</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>BrkFace</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1187</th>
      <td>Stone</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# For these three values that are clearly mismatched we will just set 
# the Mas Vnr Area to the mean area of whichever classification of Mas 
# Vnr Type it belongs to.

training_data.groupby('Mas Vnr Type')['Mas Vnr Area'].mean()
```




    Mas Vnr Type
    BrkCmn     194.000000
    BrkFace    254.824367
    None         0.000000
    Stone      230.410714
    Name: Mas Vnr Area, dtype: float64




```python
# Though there won't be any of these cases in the training set as there 
# are no rows with a Mas Vnr Type of BrkCmn and a Mas Vnr Area of 0, 
# I'm putting this here so that all my bases are covered when transfering
# all of these over to testing_data. Each of these three functions will 
# look for the not-none values in Mas Vnr Type with obviously wrong
# Mas Vnr Areas of 0, and change it to be the mean area of that specific 
# classification.

training_data.loc[(training_data['Mas Vnr Type'] == 'BrkCmn') & 
                  (training_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] \
                  = training_data.groupby('Mas Vnr Type')\
                  ['Mas Vnr Area'].mean()[0].round(0)

training_data.loc[(training_data['Mas Vnr Type'] == 'BrkFace') & 
                  (training_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] \
                  = training_data.groupby('Mas Vnr Type')\
                  ['Mas Vnr Area'].mean()[1].round(0)

training_data.loc[(training_data['Mas Vnr Type'] == 'Stone') & 
                  (training_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] \
                  = training_data.groupby('Mas Vnr Type')\
                  ['Mas Vnr Area'].mean()[3].round(0)
```


```python
training_data['Mas Vnr Area'].value_counts().sort_index()[0]
```




    1238




```python
training_data.isnull().sum().sum()
# We now have no more null values! 
```




    0



# Dropping and Creating Columns

Right off the bat the most obvious columns to drop from our dataset are Id and PID, while they may be useful for record keeping they will have no correlation with either the Sale Price or Sale Condition, making them useless for models.


```python
training_data = training_data.drop(['Id', 'PID'], axis=1)
```

Columns to drop and why (after closer inspection into data dictionary):
- **MS SubClass**: The information in this column is contained between the Year Built and the House Style columns (age of the house and number of stories)
- **Lot Shape**: I don't believe that the shape of the property will have any impact on the price.
- **Land Slope**: The information in this column is basically the same as in Land Contour but has less classificiations.
- **Garage Qual**: Has a 95% correlation with Garage Cond, is basically a repeat column.
- **Pool Area**: Only 9 of the properties have pools so just using whether or not they have a pool should be sufficient.
- **Mo Sold**: I cannot see why the month that the property is sold would have any effect on the price.
- **Garage Yr Blt**: 76% of these values are the same as the year the house was built, instead of having two heavily almost identical year columns, we can make a new column out of this - with a value of 1 if the garage was not built in the same year as the house (it'd be newer) and 0 if it was built in the same year. I will call this column Garage Newer.


```python
# Use Garage Yr Blt and Year Built to make a column with values of 1 
# if the garage is newer than the house, and 0 if they are the same age.

training_data['Garage Newer'] = training_data['Garage Yr Blt'] == \
        training_data['Year Built']
    
training_data['Garage Newer'] = \
        training_data['Garage Newer'].map({True:0, False:1})
```


```python
# Drop all of the columns mentioned above.

training_data = training_data.drop(['MS SubClass', 'Lot Shape', 
        'Land Slope', 'Garage Qual', 'Pool Area', 'Mo Sold', \
        'Garage Yr Blt'], axis=1)
```

New columns to create and why:
- **Built**: Make out of the Year Built column by splitting the years into bins, because having a scaled column of years where the lowest one is 1872 and the highest is 2010 (see below) will be very concentrated in the upper end of the values and not give much to our model. If we instead sepearate them into bins (2006 - 2010, 2001 - 2005, 1991 - 2000, etc.) and make dummy columns out of the categories, I believe this will make the information much more valuable to the model.
- **Remodeled/Additions**: Make out of the Year Remod/Add and Year Built columns. Since the year in the Year Remod/Add column will be the same as the Year Built column if there has been no remodeling or additions, you can set it to have values of 1 if the Year Built is not equal to the Year Remod/Add, and 0 if it is. This will be a simple column to signify if the property has undergone and improvements since it was originally built.
- **Multiple Exteriors**: Make out of the Exterior 1st and Exterior 2nd columns. If the entire exterior is made out of the same material the two columns will be equal to each other, but if there are two different types they will not be. My reasoning behind why this may be a good feature is that having the property made out of two different materials may be done to save money (therefore reducing the value of the property when it's time to sell), as the more expensive material could be forward facing(towards the street), with the less expensive, probably worse looking material out of sight.
- **3+ Floors**: Make out of 1st Flr SF, 2nd Flr SF, and Gr Liv Area. In ~98.4% (see below) of the properties in our training set the entire non-basement square footage (Gr Liv Area) is accounted for in the first floor and second floor. This would mean that ~1.6% of the properties have more than two floors, assuming all of the square footage numbers are correct, which they seem to be as there are no negative values in (Gr Liv Area - 1st Flr SF - 2nd Flr SF), and the smallest is 53 square feet, which is very small but could just be a small attic storage space. Can set to False to 1 for 1st Flr SF + 2nd Flr SF == Gr Liv Area, with True (when the entire square footage is accounted for in the first two floors) being 0.
- **Bathrooms**: There are currently 4 different bathroom columns: Bsmt Full Bath, Bsmt Half Bath, Full Bath, and Half Bath. I feel like just one column with all of these combined would be quite sufficient. Can simply add all of these columns up after dividing Bsmt Half Bath and Half Bath by 2, in order to account for the fact that they are in fact a 'half bath'. After creating this column we can then drop the 4 other bathroom columns.


```python
training_data['Year Built'].describe()
```




    count    2051.000000
    mean     1971.708922
    std        30.177889
    min      1872.000000
    25%      1953.500000
    50%      1974.000000
    75%      2001.000000
    max      2010.000000
    Name: Year Built, dtype: float64




```python
# Calculating the percentage of properties that have only one or two 
# floors

(training_data['1st Flr SF'] + training_data['2nd Flr SF'] == \
                 training_data['Gr Liv Area']).mean()
```




    0.98391028766455391




```python
# Third floor plus square footage of all properties where there is a 
# difference between the first and second floor square footage and the 
# total square footage.
(training_data['Gr Liv Area'] - training_data['1st Flr SF'] - \
         training_data['2nd Flr SF']).value_counts().sort_index()
```




    0       2018
    53         1
    80         3
    108        1
    114        1
    120        1
    140        1
    144        1
    156        1
    205        2
    234        1
    259        1
    312        1
    360        1
    362        1
    371        1
    384        1
    390        1
    397        1
    436        1
    450        1
    473        1
    479        1
    512        1
    513        1
    514        1
    515        1
    528        1
    572        1
    697        1
    1064       1
    dtype: int64




```python
# Making a function to read all of the years and stick them into 
# appropriate bins and running it on our training set.

def bin_year_built(df):
    year_binned = []
    for year in df['Year Built']:
        if year in list(range(2006, 2011)):
            year_binned.append('2006 - 2010')
        elif year in list(range(2001, 2006)):
            year_binned.append('2001 - 2005')
        elif year in list(range(1996, 2001)):
            year_binned.append('1996 - 2000')
        elif year in list(range(1991, 1996)):
            year_binned.append('1991 - 1995')
        elif year in list(range(1981, 1991)):
            year_binned.append('1981 - 1990')
        elif year in list(range(1971, 1981)):
            year_binned.append('1971 - 1980')
        elif year in list(range(1961, 1971)):
            year_binned.append('1961 - 1970')
        elif year in list(range(1951, 1961)):
            year_binned.append('1951 - 1960')
        elif year in list(range(1941, 1951)):
            year_binned.append('1941 - 1950')
        elif year in list(range(1931, 1941)):
            year_binned.append('1931 - 1940')
        elif year in list(range(1921, 1931)):
            year_binned.append('1921 - 1930')
        elif year in list(range(1911, 1921)):
            year_binned.append('1911 - 1920')
        elif year in list(range(1901, 1911)):
            year_binned.append('1901 - 1910')
        elif year in list(range(1891, 1901)):
            year_binned.append('1891 - 1900')
        elif year <= 1890:
            year_binned.append('1890 or Earlier')
    df['Built'] = year_binned

bin_year_built(training_data)
```


```python
# Making a column Remodeled/Additions where True signifies that there has 
# been no major work done on the house since it was built, and False means
# that there has been remodeling or additions. Then changing False to 1 
# and True to 0.
training_data['Remodeled/Additions'] = training_data['Year Built'] \
                                == training_data['Year Remod/Add']
    
training_data['Remodeled/Additions'] = \
        training_data['Remodeled/Additions'].map({True:0, False:1})
```


```python
# Making a column Multiple Exteriors where 1 means that there are two 
# different materials used on the exterior of the property and 0 means 
# that there is only one material used.
training_data['Multiple Exteriors'] = training_data['Exterior 1st'] \
                                   == training_data['Exterior 2nd']
    
training_data['Multiple Exteriors'] = \
        training_data['Multiple Exteriors'].map({True:0, False:1})
```


```python
# Making a column 3+ Floors where 1 means that the property has at least 
# 3 floors, while 0 means that it has less only 1 or 2.

training_data['3+ Floors'] = (training_data['1st Flr SF'] + \
            training_data['2nd Flr SF'] == training_data['Gr Liv Area'])

training_data['3+ Floors'] = \
        training_data['3+ Floors'].map({True:0, False:1})
```


```python
# Making a column bathrooms which adds up all of the 4 bathroom columns 
# into one value, making sure to account for the half value of 'half baths'.
training_data['Bathrooms'] = (training_data['Bsmt Half Bath'] + \
    training_data['Half Bath']) / 2 + (training_data['Bsmt Full Bath'] \
    + training_data['Full Bath'])
```

We can now drop some more columns that we will no longer need since we have created more useful columns from them:
- **Year Built**: As mentioned before a column of just years will not be very useful in a model as they will just be read as values in a very tight range.
- **Year Remod/Add**: Same as Year Built, year values are not great information in a model.
- **Exterior 2nd**: Since ~84.5% of these values are the exact same as Exterior 1st because most properties only have one type of exterior, these columns are very highly correlated and are thus bad for a model.
- **1st Flr SF** and **2nd Flr SF**: The size of individual floors should not be great indicators on their own, just having the total square footage in Gr Liv Area should be quite sufficient.
- **Bsmt Half Bath**, **Bsmt Full Bath**, **Half Bath** and **Full Bath**: We have now combined all of these into one Bathrooms column and should have no need for all of these individual elements.


```python
(training_data['Exterior 1st'] == training_data['Exterior 2nd']).mean()
```




    0.84495368113115554




```python
training_data = training_data.drop(['Year Built', 'Year Remod/Add', \
        'Exterior 2nd', '1st Flr SF', '2nd Flr SF', 'Bsmt Half Bath', \
        'Bsmt Full Bath', 'Half Bath', 'Full Bath'], axis=1)
```

# Make Same Changes for Testing Set

In order to make our predictions on Kaggle's testing data set, I will need to make all of the same changes to it as I have to the training data. Since I've coded all of the changes very generally, I should be able to easily make the changes required by copying all the ones we made to the training set above and changing the training_data variable to testing_data.


```python
testing_data = testing_data.drop('Lot Frontage', axis=1)

testing_data['Alley'].fillna('No Alley', inplace=True)
testing_data['Fireplace Qu'].fillna('No Fireplace', inplace=True)
testing_data['Pool QC'].fillna('No Pool', inplace=True)
testing_data['Fence'].fillna('No Fence', inplace=True)
testing_data['Misc Feature'].fillna('No Misc Feature', inplace=True)
```


```python
testing_data.loc[((testing_data['Garage Type'].notnull()) & 
        (testing_data['Garage Qual'].isnull()) & 
        (testing_data['Garage Finish'].isnull()) & 
        (testing_data['Garage Yr Blt'].isnull()) & 
        (testing_data['Garage Yr Blt'].isnull()), 'Garage Type')] = np.nan 
```


```python
testing_data.loc[(testing_data['Garage Type'].isnull() & 
        testing_data['Garage Cars'].isnull()), 'Garage Cars'] = 0

testing_data.loc[(testing_data['Garage Type'].isnull() & 
        testing_data['Garage Area'].isnull()), 'Garage Area'] = 0
```


```python
testing_data['Garage Type'].fillna('No Garage', inplace=True)
testing_data['Garage Finish'].fillna('No Garage', inplace=True)
testing_data['Garage Qual'].fillna('No Garage', inplace=True)
testing_data['Garage Cond'].fillna('No Garage', inplace=True)
testing_data['Garage Yr Blt'].fillna('No Garage', inplace=True)
```


```python
testing_data.loc[(training_data['Bsmt Qual'].notnull() 
    & testing_data['Bsmt Exposure'].isnull()), 'Bsmt Exposure'] = 'No'
```


```python
testing_data.loc[(training_data['BsmtFin Type 1'].notnull() & 
    testing_data['BsmtFin Type 2'].isnull()), 'BsmtFin Type 2'] = 'Unf'
```


```python
testing_data[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']] = \
testing_data[['Bsmt Qual', 'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', 'BsmtFin Type 2']].fillna('No Basement')
```


```python
testing_data.loc[((testing_data['Bsmt Qual'] == 'No Basement') & 
            (testing_data['Bsmt Cond'] == 'No Basement') & 
            (testing_data['Bsmt Exposure'] == 'No Basement') & 
            (testing_data['BsmtFin Type 1'] == 'No Basement') &
            (testing_data['BsmtFin Type 2'] == 'No Basement'), \
                                  'BsmtFin SF 1')] = 0

testing_data.loc[((testing_data['Bsmt Qual'] == 'No Basement') & 
            (testing_data['Bsmt Cond'] == 'No Basement') & 
            (testing_data['Bsmt Exposure'] == 'No Basement') & 
            (testing_data['BsmtFin Type 1'] == 'No Basement') &
            (testing_data['BsmtFin Type 2'] == 'No Basement'), \
                                  'BsmtFin SF 2')] = 0

testing_data.loc[((training_data['Bsmt Qual'] == 'No Basement') & 
            (testing_data['Bsmt Cond'] == 'No Basement') &
            (testing_data['Bsmt Exposure'] == 'No Basement') & 
            (testing_data['BsmtFin Type 1'] == 'No Basement') &
            (testing_data['BsmtFin Type 2'] == 'No Basement'), \
                                  'Bsmt Unf SF')] = 0

testing_data.loc[((testing_data['Bsmt Qual'] == 'No Basement') & 
                  (testing_data['Bsmt Cond'] == 'No Basement') &
                  (testing_data['Bsmt Exposure'] == 'No Basement') & 
                  (testing_data['BsmtFin Type 1'] == 'No Basement') &
                  (testing_data['BsmtFin Type 2'] == 'No Basement'), \
                                  'Total Bsmt SF')] = 0
```


```python
testing_data['Bsmt Half Bath'].fillna(value = 0, inplace=True)
testing_data['Bsmt Full Bath'].fillna(value = 0, inplace=True)
```


```python
testing_data['Mas Vnr Type'].fillna(value='None', inplace=True)
testing_data['Mas Vnr Area'].fillna(value=0, inplace=True)
```


```python
testing_data.loc[(testing_data['Mas Vnr Type'] == 'None') & 
        (testing_data['Mas Vnr Area'] == 1), 'Mas Vnr Area'] = 0
```


```python
testing_data.loc[(testing_data['Mas Vnr Type'] == 'BrkCmn') & 
    (testing_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] = \
    testing_data.groupby('Mas Vnr Type')['Mas Vnr Area'].mean()[0].round(0)

testing_data.loc[(testing_data['Mas Vnr Type'] == 'BrkFace') & 
    (testing_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] = \
    testing_data.groupby('Mas Vnr Type')['Mas Vnr Area'].mean()[1].round(0)

testing_data.loc[(testing_data['Mas Vnr Type'] == 'Stone') & 
    (testing_data['Mas Vnr Area'] == 0), 'Mas Vnr Area'] = \
    testing_data.groupby('Mas Vnr Type')['Mas Vnr Area'].mean()[3].round(0)
```

### Dropping and Creating Columns

Once again we will just copy the steps we followed when dropping and creating columns for the training set, with changing the variable name to testing_data. Since the datasets both have the same columns (except for the extra 2 target variables in training_data, which we haven't touched yet), all of the changes should work just fine on the testing_data set.


```python
# Before we drop the testing_data Id, we need to extract it into it's own
# dataframe for purposes of submission to Kaggle later.
testid = testing_data['Id']
testing_data = testing_data.drop(['Id', 'PID'], axis=1)
```


```python
testing_data['Garage Newer'] = testing_data['Garage Yr Blt'] == \
                               testing_data['Year Built']
    
testing_data['Garage Newer'] = \
            testing_data['Garage Newer'].map({True:0, False:1})
```


```python
testing_data = testing_data.drop(['MS SubClass', 'Lot Shape', 
        'Land Slope', 'Garage Qual', 'Pool Area', 'Mo Sold', \
        'Garage Yr Blt'], axis=1)
```


```python
# We already made a function for binning the Year Built column, we can 
# just call it again on testing_data.
bin_year_built(testing_data)
```


```python
testing_data['Remodeled/Additions'] = testing_data['Year Built'] == \
                                      testing_data['Year Remod/Add']
    
testing_data['Remodeled/Additions'] = \
            testing_data['Remodeled/Additions'].map({True:0, False:1})
```


```python
testing_data['Multiple Exteriors'] = testing_data['Exterior 1st'] == \
                                     testing_data['Exterior 2nd']
    
testing_data['Multiple Exteriors'] = \
            testing_data['Multiple Exteriors'].map({True:0, False:1})
```


```python
testing_data['3+ Floors'] = (testing_data['1st Flr SF'] + 
                             testing_data['2nd Flr SF'] == \
                             testing_data['Gr Liv Area'])

testing_data['3+ Floors'] = \
            testing_data['3+ Floors'].map({True:0, False:1})
```


```python
testing_data['Bathrooms'] = (testing_data['Bsmt Half Bath'] + \
        testing_data['Half Bath']) / 2 + \
        (testing_data['Bsmt Full Bath'] + testing_data['Full Bath'])
```


```python
testing_data = testing_data.drop(['Year Built', 'Year Remod/Add', \
        'Exterior 2nd', '1st Flr SF', '2nd Flr SF', 'Bsmt Half Bath', \
        'Bsmt Full Bath', 'Half Bath', 'Full Bath'], axis=1)
```

# Setting Targets and Aligning our Datasets

We can now extract our two target variables. SalePrice is the regression target, and Sale Condition is the classification target.


```python
y_regression = training_data['SalePrice']
y_classification = training_data['Sale Condition']
```

After the targets have been extracted into their own Series, we can then drop them from training_data so that training_data and testing_data's columns now line up.


```python
training_data = \
    training_data.drop(['SalePrice', 'Sale Condition'], axis=1)

print(training_data.shape, testing_data.shape)
```


```python
print(training_data.shape, testing_data.shape)
```

    (2051, 67) (879, 67)


# Creating Numerical Data From Categorical Strings

### Converting Dual-Categories into Binary

The following columns have only two different possible classifications and can thus be changed into 1s and 0s. In both of these cases the most common will be 0, with the least common being 1.


```python
training_data['Street'] = \
    training_data['Street'].map({'Pave':0, 'Grvl':1})
training_data['Central Air'] = \
    training_data['Central Air'].map({'Y':0, 'N':1})

testing_data['Street'] = \
    testing_data['Street'].map({'Pave':0, 'Grvl':1})
testing_data['Central Air'] = \
    testing_data['Central Air'].map({'Y':0, 'N':1})
```

### Dummy Columns

Since so much of our data is currently in categorical strings, in order to use it in our models we will have to create dummy values out of those columns. The columns I will create dummys out of, along with which of the dummy columns we will drop from the dataset after they are created (in almost every case it is the most common classification) are as follows:
- **MS Zoning**: Will drop RL (Residential Low Density)
- **Alley**: Will drop No Alley
- **Land Contour**: Will drop Lvl (Near Flat/Level)
- **Utilities**: Will drop AllPub (All Public Utilities (Electricity, Gas and Water))
- **Lot Config**: Will drop Inside (Inside Lot)
- **Neighborhood**: Will drop NAmes (North Ames)
- **Condition 1**: Will drop Norm (Normal)
- **Condition 2**: Will drop Norm (Normal)
- **Bldg Type**: Will drop 1Fam (Single-family detached)
- **House Style**: Will drop 1Story (One story)
- **Overall Qual**: Will drop 5 (Average)
- **Overall Cond**: Will drop 5 (Average)
- **Roof Style**: Will drop Gable 
- **Roof Matl**: Will drop CompShg (Standard (Composite) Shingle)
- **Exterior 1st**: Will drop VinylSd (Vinyl Sliding)
- **Mas Vnr Type**: Will drop None
- **Exter Qual**: Will drop TA (Typical/Average)
- **Exter Cond**: Will drop TA (Typical/Average)
- **Foundation**: Will drop PConc (Poured Concrete)
- **Bsmt Qual**: Will drop TA (Typical/Average)
- **Bsmt Cond**: Will drop TA (Typical/Average)
- **Bsmt Exposure**: Will drop No (No Exposure)
- **BsmtFin Type 1**: Will drop GLQ (Good Living Quarters)
- **BsmtFin Type 2**: Will drop Unf (Unfinished)
- **Heating**: Will drop GasA (Gas forced warm air furnace)
- **Heating QC**: Will drop Ex (Excellent)
- **Electrical**: Will drop SBrkr (Standard Circuit Breakers & Romex)
- **Kitchen Qual**: Will drop TA (Typical/Average)
- **Functional**: Will drop Typ (Typical Funcionality)
- **Fireplace Qu**: Will drop No Fireplace
- **Garage Type**: Will drop Attchd (Attached to home)
- **Garage Finish**: Will drop Unf (Unfinished)
- **Garage Cond**: Will drop TA (Typical/Average)
- **Paved Drive**: Will drop Y (Paved)
- **Pool QC**: Will drop No Pool
- **Fence**: Will drop No Fence
- **Misc Feature**: Will drop No Misc Feature
- **Yr Sold**: Will drop 2007
- **Sale Type**: Will drop WD (Warranty Deed - Conventional)
- **Built**: Will drop 2001 - 2005


```python
cols_to_dummy = ['MS Zoning', 'Alley', 'Land Contour', 'Utilities', \
                 'Lot Config', 'Neighborhood', 'Condition 1', \
                'Condition 2', 'Bldg Type', 'House Style', \
                'Overall Qual', 'Overall Cond', 'Roof Style', \
                'Roof Matl', 'Exterior 1st', 'Mas Vnr Type', \
                'Exter Qual', 'Exter Cond', 'Foundation', 'Bsmt Qual', \
                'Bsmt Cond', 'Bsmt Exposure', 'BsmtFin Type 1', \
                'BsmtFin Type 2', 'Heating', 'Heating QC', 'Electrical', \
                'Kitchen Qual', 'Functional', 'Fireplace Qu', \
                'Garage Type', 'Garage Finish', 'Garage Cond', \
                'Paved Drive', 'Pool QC', 'Fence', 'Misc Feature', \
                'Yr Sold', 'Sale Type', 'Built']
```


```python
testing_data['Electrical'].isnull().sum()
```




    1



The make dummies function below would not run because it was encountering an error with mixed group types of str and float. After checking that there were no float columns being used in it (the list above), I noticed there was a null value in the Electrical column (I also learned np.nan is considered a float), I then changed that to the most common value in the column as an easy way to impute it.


```python
testing_data['Electrical'].fillna\
    (testing_data['Electrical'].value_counts().index[0], inplace=True)
```


```python
testing_data['Electrical'].isnull().sum()
```




    0



From diving deeper into the data I realized that I would run into problems making dummies for both the testing and training sets as there was a value in the testing set that did not appear in the training set. In order to get around this I used LabelBinarizer fit only on the training data, so that when it was transformed with the testing data it ignored any dummies that did not already exist. The function then also adds those new dummies back to the original data frames and drops the parent columns.


```python
def make_dummies(df_training, df_testing, dummy_cols):
    for col in dummy_cols:
        lb.fit(df_training[col])
        columns = [str(col) + '_' + str(x) for x in lb.classes_]
        transformed_set = pd.DataFrame(lb.transform(df_training[col]), columns = columns)
        df_training = pd.concat([df_training.reset_index(drop=True), transformed_set], axis =1)
        test_transformed_set = pd.DataFrame(lb.transform(df_testing[col]), columns = columns)
        df_testing = pd.concat([df_testing.reset_index(drop=True), test_transformed_set], axis =1)
    df_training = df_training.drop(dummy_cols, axis =1)
    df_testing = df_testing.drop(dummy_cols, axis =1)
    return df_training, df_testing
```


```python
lb = LabelBinarizer()
training_data, testing_data = \
        make_dummies(training_data, testing_data, cols_to_dummy)
```


```python
training_data.shape
```




    (2051, 301)




```python
testing_data.shape
```




    (879, 301)



# Scaling and Splitting our Data

Now that all both our training and testing data sets have been completely cleaned without taking too much of a look into the testing set, we can scale our predictors and then use train test split to get a set of data we can test our models with. I am aware this is not the optimal way to train test split as it should be done before I touch any of the data at all, but this way is much easier and hopefully it won't have too much of a negative impact on our model. For next time I would like to try it the optimal way but for now running all of my data cleaning methods on two datasets was more than enough.


```python
ss = StandardScaler()
X = ss.fit_transform(training_data)
```


```python
ss = StandardScaler()
testX = ss.fit_transform(testing_data)
```


```python
X_train_regression, X_test_regression, \
y_train_regression, y_test_regression = \
        train_test_split(X, y_regression, test_size=0.30)
```

Because the Kaggle submission is only looking for whether or not the Sale condition is abnormal or not, we will have to change all non-abnormal values to 0 and abnormal values to 1.


```python
y_classification.value_counts()
```




    Normal     1696
    Partial     164
    Abnorml     132
    Family       29
    Alloca       19
    AdjLand      11
    Name: Sale Condition, dtype: int64




```python
y_classification = \
    y_classification.apply(lambda x: 1 if x == 'Abnorml' else 0)
```


```python
y_classification.value_counts()
```




    0    1919
    1     132
    Name: Sale Condition, dtype: int64




```python
X_train_classification, X_test_classification, \
y_train_classification, y_test_classification = \
        train_test_split(X, y_classification, test_size=0.30)
```

# Modeling!

## Regression

We will start out with training the base regression models on our training data, see which one fares best with a cross_val_score on the testing data, and then dig deeper into that one with a grid search and possibly feature selection if necessary.

### Linear


```python
lin_reg = LinearRegression()
lin_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(lin_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    -5.21127595219e+24


### Lasso


```python
lasso_reg = Lasso()
lasso_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(lasso_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


    0.410594788002


    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/coordinate_descent.py:491: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Fitting data with very small alpha may cause precision problems.
      ConvergenceWarning)


### Ridge


```python
ridge_reg = Ridge()
ridge_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(ridge_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    0.577867803424


### Elastic Net


```python
en_reg = ElasticNet()
en_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(en_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    0.822984908117


### Logistic


```python
log_reg = LogisticRegression()
log_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(log_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)


    0.0175376613577


### Random Forest Regressor


```python
rf_reg = RandomForestRegressor()
rf_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(rf_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    0.816822897533


### Gradient Boosting Regressor


```python
gb_reg = GradientBoostingRegressor()
gb_reg.fit(X_train_regression, y_train_regression)

print(cross_val_score(gb_reg, X_test_regression, \
                      y_test_regression, cv=5).mean())
```

    0.842365991641


### Grid Searches of Best Model


```python
param = {
    'n_estimators': [i for i in range(1900,2101) if i % 50 == 0],
    'max_features' : ['sqrt', 'auto', None],
    'learning_rate' : [0.03, 0.035, 0.04]
    }

est = GradientBoostingRegressor()

reg_clf = GridSearchCV(est, param_grid=param, verbose=1, n_jobs=-1)

reg_clf.fit(X_train_regression, y_train_regression)
print(reg_clf.best_params_)
print(reg_clf.best_score_)
```

    Fitting 3 folds for each of 45 candidates, totalling 135 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:  1.6min
    [Parallel(n_jobs=-1)]: Done 135 out of 135 | elapsed:  6.0min finished


    {'learning_rate': 0.04, 'max_features': 'sqrt', 'n_estimators': 2000}
    0.908152362991



```python
predictions = reg_clf.predict(X_test_regression)
```


```python
plt.scatter(y_test_regression, predictions)
plt.plot((min(y_test_regression),max(y_test_regression)),\
         (min(y_test_regression),max(y_test_regression)),c='k')
plt.ylabel('Gradient Boosting Regressor Predicted Value')
plt.xlabel('Actual Values')
plt.title ('Iowa property prices, R2={}'.format\
           (reg_clf.score(X_test_regression,y_test_regression)));
```


![png](/images/ames-housing_files/ames-housing_156_0.png)



```python
gbr_residuals = y_test_regression - predictions
```


```python
plt.scatter(y_test_regression, gbr_residuals)
plt.axhline(0)
plt.ylabel('Residuals from GBR Predicted Value')
plt.xlabel('Actual Value of Property Sold')
plt.title ('Residuals from Gradient Boosting Regressor');
```


![png](/images/ames-housing_files/ames-housing_158_0.png)



```python
plt.hist(gbr_residuals)
plt.xlabel('Actual Value - Predicted Value')
plt.title ('Residuals from Gradient Boosting Regressor');
```


![png](/images/ames-housing_files/ames-housing_159_0.png)


These graphs make it quite clear that while the model is predicting the sale price quite accurately in general, it is definitely better in some places than others. It does not do well on extremely low values (less than 30,000), or on higher values (greater than 350,000), but it is very accurate in the middle area, where the vast majority of the properties lie. With the exception of a few outliers which I may try to look into individually in the future, I am quite happy with the performance of this model.

## Classification


```python
y_classification.value_counts()
```




    0    1919
    1     132
    Name: Sale Condition, dtype: int64




```python
y_classification.value_counts()[0] / \
    (y_classification.value_counts()[0] + \
     y_classification.value_counts()[1])
```




    0.93564115065821551



This model seemed much more difficult to do well on than the regression model. Because the baseline is so high (93.6%), it made my best performing models barely any better than the ones that just predicted that all sales were not abnormal. It appears as if predicting whether or not a sale was abnormal is not an easy task.

### Random Forest Classifier


```python
rfc = RandomForestClassifier()
rfc.fit(X_train_classification, y_train_classification)

print(cross_val_score(rfc, X_test_classification, \
            y_test_classification, cv=5).mean())
```

    0.939942302649



```python
predicted = rfc.predict(X_test_classification)
con_mat = confusion_matrix(y_test_classification, predicted)

confusion = pd.DataFrame(con_mat, index=['not_abnormal', 'is_abormal'], \
              columns=['predicted_not_abnormal', 'predicted_abnormal'])

confusion
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_not_abnormal</th>
      <th>predicted_abnormal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not_abnormal</th>
      <td>578</td>
      <td>2</td>
    </tr>
    <tr>
      <th>is_abormal</th>
      <td>35</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### Gradient Boosting Classifier


```python
gbc = GradientBoostingClassifier()
gbc.fit(X_train_classification, y_train_classification)

print(cross_val_score(gbc, X_test_classification, \
            y_test_classification, cv=5).mean())
```

    0.948059270915



```python
predicted = gbc.predict(X_test_classification)
con_mat = confusion_matrix(y_test_classification, predicted)

confusion = pd.DataFrame(con_mat, index=['not_abnormal', 'is_abormal'], \
              columns=['predicted_not_abnormal', 'predicted_abnormal'])

confusion
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_not_abnormal</th>
      <th>predicted_abnormal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not_abnormal</th>
      <td>570</td>
      <td>10</td>
    </tr>
    <tr>
      <th>is_abormal</th>
      <td>29</td>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



It's quite clear from looking at the confusion matrix that while the Random Forest Classifier and Gradient Boosting Classifier have similar accuracies, they are quite different in the outcome. Random Forest Classifier only predicted one abnormal sale, albeit correctly, while Gradient Boosting Classifier predicted 17 abnormal sales with only 7 being correct. For further evaluation I'm going to use Gradient Boosting Classifier.

### Grid Searches of Best Model


```python
param = {
    'n_estimators': [i for i in range(300,401) if i % 10 == 0],
    'max_features' : ['sqrt'],
    'learning_rate' : [0.01, 0.02, 0.03]
    }

gbc = GradientBoostingClassifier()

class_clf = GridSearchCV(gbc, param_grid=param, verbose=1, n_jobs=-1)

class_clf.fit(X_train_classification, y_train_classification)
print(class_clf.best_params_)
print(class_clf.best_score_)
```

    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    3.7s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    9.2s finished


    {'learning_rate': 0.02, 'max_features': 'sqrt', 'n_estimators': 310}
    0.934494773519



```python
print(cross_val_score(class_clf, X_test_classification, \
            y_test_classification, cv=5).mean())
```

    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.7s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    4.2s finished


    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.5s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    4.0s finished


    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.6s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    4.1s finished


    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.6s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    4.1s finished


    Fitting 3 folds for each of 33 candidates, totalling 99 fits


    [Parallel(n_jobs=-1)]: Done  34 tasks      | elapsed:    1.6s
    [Parallel(n_jobs=-1)]: Done  99 out of  99 | elapsed:    4.1s finished


    0.938316286389



```python
predicted = class_clf.predict(X_test_classification)
con_mat = confusion_matrix(y_test_classification, predicted)

confusion = pd.DataFrame(con_mat, index=['not_abnormal', 'is_abormal'], \
              columns=['predicted_not_abnormal', 'predicted_abnormal'])

confusion
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>predicted_not_abnormal</th>
      <th>predicted_abnormal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>not_abnormal</th>
      <td>575</td>
      <td>5</td>
    </tr>
    <tr>
      <th>is_abormal</th>
      <td>33</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



For some reason I was completely unable to get a better cross val score on the test set with any of the hyperparameters from many different grid searches. However, it does affect how the model predicts, by now making it predict abnormal less.

To see how adjusting the threshold of the predicted probability impacts the weight of the false positive rate vs. the true positive, we can build an ROC curve using the predicted probabilities.


```python
y_pp = pd.DataFrame(class_clf.predict_proba(X_test_classification), \
                    columns=['not_abnormal_pp', 'abnormal_pp'])

fpr, tpr, thresh = roc_curve(y_test_classification, y_pp.abnormal_pp)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=[8,8])
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc, 
         linewidth=4)
plt.plot([0, 1], [0, 1], 'k--', linewidth=4)
plt.xlim([-0.05, 1.0])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=18)
plt.ylabel('True Positive Rate', fontsize=18)
plt.title('Receiver operating characteristic for abnormal sale prediction', 
          fontsize=18)
plt.legend(loc="lower right")
plt.show()
```


![png](/images/ames-housing_files/ames-housing_177_0.png)


From the ROC curve we can see that changing the threshold does not really give us much of an increase in the precision of our model. For next time I'd like to try using something like SMOTE (Synthetic Minority Over-sampling Technique) which over-samples the minority class, in this case it would be abnormal house sales, by creating synthetic examples of that class in order to try better train your model to predict them.

# Feature Importance

Since so much time and energy was spent on feature engineering for this project, it would be a shame if we didn't even look at how well the features ended up being for our models!

### Regression


```python
reg_clf.best_params_
```




    {'learning_rate': 0.04, 'max_features': 'sqrt', 'n_estimators': 2000}




```python
best_gbr = GradientBoostingRegressor(learning_rate=0.04, 
                    max_features='sqrt', n_estimators=2000)
best_gbr.fit(X_train_regression, y_train_regression)

# Looking at our strongest coefficients for the optimized through
# GridSearch Gradient Boosting Regressor model

reg_feature_importance = pd.DataFrame(best_gbr.feature_importances_,
                index=training_data.columns, columns=['coefficients'])

reg_feature_importance.sort_values\
                        (by='coefficients', ascending=False).head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Gr Liv Area</th>
      <td>0.057992</td>
    </tr>
    <tr>
      <th>Lot Area</th>
      <td>0.044143</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>0.039120</td>
    </tr>
    <tr>
      <th>BsmtFin SF 1</th>
      <td>0.037801</td>
    </tr>
    <tr>
      <th>Bsmt Unf SF</th>
      <td>0.033328</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>0.033256</td>
    </tr>
    <tr>
      <th>Mas Vnr Area</th>
      <td>0.025700</td>
    </tr>
    <tr>
      <th>Wood Deck SF</th>
      <td>0.019216</td>
    </tr>
    <tr>
      <th>Bathrooms</th>
      <td>0.018953</td>
    </tr>
    <tr>
      <th>TotRms AbvGrd</th>
      <td>0.016822</td>
    </tr>
    <tr>
      <th>Bedroom AbvGr</th>
      <td>0.013359</td>
    </tr>
    <tr>
      <th>Enclosed Porch</th>
      <td>0.012368</td>
    </tr>
    <tr>
      <th>Open Porch SF</th>
      <td>0.012108</td>
    </tr>
    <tr>
      <th>Garage Cars</th>
      <td>0.011885</td>
    </tr>
    <tr>
      <th>Screen Porch</th>
      <td>0.010778</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>0.010671</td>
    </tr>
    <tr>
      <th>Kitchen Qual_TA</th>
      <td>0.009229</td>
    </tr>
    <tr>
      <th>Neighborhood_GrnHill</th>
      <td>0.008695</td>
    </tr>
    <tr>
      <th>Heating QC_Ex</th>
      <td>0.007653</td>
    </tr>
    <tr>
      <th>Overall Cond_4</th>
      <td>0.007068</td>
    </tr>
  </tbody>
</table>
</div>



This seems to be just about what one could expect to see in what most contributes to a houses sale price, most of the top 10 coefficients all have to do with square footage, and the ones that aren't are the numbers of bedrooms or bathrooms.


```python
# Want to see how many of our 301 predictors wound up being useful
# at all for the model

non_zero_coefs = reg_feature_importance['coefficients'] != 0

len(reg_feature_importance[non_zero_coefs])
```




    273



273 of the 301 original predictors have a value of non-zero, which means that only 28 were not useful at all for the model. I will now just quickly check how the features that I created did, and hopefully they are not among the 28! 


```python
reg_feature_importance.loc['3+ Floors']
```




    coefficients    0.000475
    Name: 3+ Floors, dtype: float64




```python
reg_feature_importance.loc['Remodeled/Additions']
```




    coefficients    0.004548
    Name: Remodeled/Additions, dtype: float64




```python
reg_feature_importance.loc['Multiple Exteriors']
```




    coefficients    0.002974
    Name: Multiple Exteriors, dtype: float64




```python
reg_feature_importance.loc['Bathrooms']
```




    coefficients    0.014425
    Name: Bathrooms, dtype: float64




```python
built_bins = ['Built_2006 - 2010', 'Built_2001 - 2005', \
              'Built_1996 - 2000', 'Built_1991 - 1995', \
              'Built_1981 - 1990', 'Built_1971 - 1980', \
              'Built_1961 - 1970', 'Built_1951 - 1960', \
              'Built_1941 - 1950', 'Built_1931 - 1940', \
              'Built_1921 - 1930', 'Built_1911 - 1920', \
              'Built_1901 - 1910', 'Built_1891 - 1900', \
              'Built_1890 or Earlier']
```


```python
reg_feature_importance.loc[built_bins]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Built_2006 - 2010</th>
      <td>0.006992</td>
    </tr>
    <tr>
      <th>Built_2001 - 2005</th>
      <td>0.002948</td>
    </tr>
    <tr>
      <th>Built_1996 - 2000</th>
      <td>0.003454</td>
    </tr>
    <tr>
      <th>Built_1991 - 1995</th>
      <td>0.003367</td>
    </tr>
    <tr>
      <th>Built_1981 - 1990</th>
      <td>0.002348</td>
    </tr>
    <tr>
      <th>Built_1971 - 1980</th>
      <td>0.001238</td>
    </tr>
    <tr>
      <th>Built_1961 - 1970</th>
      <td>0.002220</td>
    </tr>
    <tr>
      <th>Built_1951 - 1960</th>
      <td>0.002521</td>
    </tr>
    <tr>
      <th>Built_1941 - 1950</th>
      <td>0.002301</td>
    </tr>
    <tr>
      <th>Built_1931 - 1940</th>
      <td>0.002893</td>
    </tr>
    <tr>
      <th>Built_1921 - 1930</th>
      <td>0.001838</td>
    </tr>
    <tr>
      <th>Built_1911 - 1920</th>
      <td>0.001791</td>
    </tr>
    <tr>
      <th>Built_1901 - 1910</th>
      <td>0.001017</td>
    </tr>
    <tr>
      <th>Built_1891 - 1900</th>
      <td>0.001880</td>
    </tr>
    <tr>
      <th>Built_1890 or Earlier</th>
      <td>0.004394</td>
    </tr>
  </tbody>
</table>
</div>



So all of the features created in the feature engineering process ended up being at least somewhat useful to the regression model, including every single one of the year built bins. Of all of them, bathrooms was the strongest predictor by quite a bit, with whether or not the property was built in the most recent bin of years or if it has been remodeled also being decent predictors.

### Classification


```python
class_clf.best_params_
```




    {'learning_rate': 0.02, 'max_features': 'sqrt', 'n_estimators': 310}




```python
best_gbc = GradientBoostingClassifier(learning_rate=0.02, 
                        max_features='sqrt', n_estimators=310)
best_gbc.fit(X_train_classification, y_train_classification)

class_feature_importance = pd.DataFrame(best_gbc.feature_importances_, 
            index=training_data.columns, columns=['coefficients'])

class_feature_importance.sort_values(by='coefficients', \
                                     ascending=False).head(20)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Sale Type_COD</th>
      <td>0.084181</td>
    </tr>
    <tr>
      <th>Condition 2_Artery</th>
      <td>0.048354</td>
    </tr>
    <tr>
      <th>BsmtFin SF 1</th>
      <td>0.037307</td>
    </tr>
    <tr>
      <th>Central Air</th>
      <td>0.034883</td>
    </tr>
    <tr>
      <th>Exterior 1st_Stone</th>
      <td>0.033167</td>
    </tr>
    <tr>
      <th>Open Porch SF</th>
      <td>0.028233</td>
    </tr>
    <tr>
      <th>Functional_Sal</th>
      <td>0.027796</td>
    </tr>
    <tr>
      <th>MS Zoning_C (all)</th>
      <td>0.026157</td>
    </tr>
    <tr>
      <th>Mas Vnr Type_BrkCmn</th>
      <td>0.023661</td>
    </tr>
    <tr>
      <th>Sale Type_Oth</th>
      <td>0.020128</td>
    </tr>
    <tr>
      <th>Garage Area</th>
      <td>0.020113</td>
    </tr>
    <tr>
      <th>Bsmt Unf SF</th>
      <td>0.018946</td>
    </tr>
    <tr>
      <th>Lot Area</th>
      <td>0.016717</td>
    </tr>
    <tr>
      <th>Yr Sold_2007</th>
      <td>0.016287</td>
    </tr>
    <tr>
      <th>Gr Liv Area</th>
      <td>0.014659</td>
    </tr>
    <tr>
      <th>Condition 1_RRNn</th>
      <td>0.013891</td>
    </tr>
    <tr>
      <th>Total Bsmt SF</th>
      <td>0.012612</td>
    </tr>
    <tr>
      <th>Sale Type_WD</th>
      <td>0.011097</td>
    </tr>
    <tr>
      <th>Yr Sold_2006</th>
      <td>0.010722</td>
    </tr>
    <tr>
      <th>Enclosed Porch</th>
      <td>0.010522</td>
    </tr>
  </tbody>
</table>
</div>



The feature that is the strongest predictor of a property sale being abnormal or not is the sale being made for cash. Makes sense when you think about it, but still an interesting discovery.


```python
non_zero_coefs = class_feature_importance['coefficients'] != 0

len(feature_importance[non_zero_coefs])
```




    225



76 of the 301 predictors got zeroed out in the classification model. Let's again see how the created features did.


```python
class_feature_importance.loc['3+ Floors']
```




    coefficients    0.001659
    Name: 3+ Floors, dtype: float64




```python
class_feature_importance.loc['Remodeled/Additions']
```




    coefficients    0.000868
    Name: Remodeled/Additions, dtype: float64




```python
class_feature_importance.loc['Multiple Exteriors']
```




    coefficients    0.001223
    Name: Multiple Exteriors, dtype: float64




```python
class_feature_importance.loc['Bathrooms']
```




    coefficients    0.006495
    Name: Bathrooms, dtype: float64




```python
class_feature_importance.loc[built_bins]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>coefficients</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Built_2006 - 2010</th>
      <td>0.002324</td>
    </tr>
    <tr>
      <th>Built_2001 - 2005</th>
      <td>0.001090</td>
    </tr>
    <tr>
      <th>Built_1996 - 2000</th>
      <td>0.000745</td>
    </tr>
    <tr>
      <th>Built_1991 - 1995</th>
      <td>0.000681</td>
    </tr>
    <tr>
      <th>Built_1981 - 1990</th>
      <td>0.000120</td>
    </tr>
    <tr>
      <th>Built_1971 - 1980</th>
      <td>0.001721</td>
    </tr>
    <tr>
      <th>Built_1961 - 1970</th>
      <td>0.001754</td>
    </tr>
    <tr>
      <th>Built_1951 - 1960</th>
      <td>0.001452</td>
    </tr>
    <tr>
      <th>Built_1941 - 1950</th>
      <td>0.002421</td>
    </tr>
    <tr>
      <th>Built_1931 - 1940</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Built_1921 - 1930</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Built_1911 - 1920</th>
      <td>0.001800</td>
    </tr>
    <tr>
      <th>Built_1901 - 1910</th>
      <td>0.001569</td>
    </tr>
    <tr>
      <th>Built_1891 - 1900</th>
      <td>0.000097</td>
    </tr>
    <tr>
      <th>Built_1890 or Earlier</th>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



So none of the created features were very great predictors of a sale being abnormal or not, but only a few of the much older year built bins got zeroed out.
