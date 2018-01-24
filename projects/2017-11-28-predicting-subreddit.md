---
layout: page
title: Predicting Subreddits
date: 2017-11-28
published: true
---

This dataset contains all of the posts made to a specific set of programming-oriented subreddits in the month of December. We will make a model that attempts to accurately predict which subreddit the post was made to. I received this data from an instructor who provided it for an open-ended project.


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
```

## Quick Fix to Dataset

The data has (supposedly) been subset to include only self-posts (posts where the the writer has submitted text instead of a link). 

However, when I was taking a deeper dive into the data, I noticed that not only self-posts were included within it. I also noticed that all of the non-self posts that had gotten through were deleted or removed from the subreddit. 

When a post is deleted (by the user) or removed (by a moderator/admin), it gives the text of the post, which would not have existed before removal, a value of [deleted] or [removed]. I am assuming that this is what caused it to slip by whatever filter there was as the column 'selftext' was no longer empty like it should be. 

Since I will be relying mostly on Natural Language Processing for this model, and posts which are deleted or removed no longer have any selftext to process, I will actually be dropping all entries with a 'selftext' value of [deleted] or [removed]. This will solve both of these problems and make the model comparatively more reliant on the selftext and less reliant on the title (which remains even when a post is deleted or removed). To do this quick fix I did:


```python
df = pd.read_csv('reddit_posts.csv')
```




```python
df.shape
```




    (26688, 53)




```python
df['selftext'][df['selftext'] == '[removed]'].count() + \
        df['selftext'][df['selftext'] == '[deleted]'].count()
```




    8480



There are a total of 8480 posts or about a third of our dataset that have either been deleted or removed. This will severely impact our Natural Language Processing later on, as all of these entries will have 0 values for every column created from the NLP on the 'selftext' column. This just further confirms my prior hypothesis that these should be deleted from the dataset. This would make it so that my model is bad at predicting the subreddit of a deleted or removed post, as the model would rely less on the 'title' processed columns than the 'selftext' processed columns, but it should be much better at guessing the subreddit of posts with actual 'selftext' values.


```python
# Changing deleted or removed posts to null values

df['selftext'] = df['selftext'].apply(lambda x: np.nan if x == \
                    '[deleted]' or x == '[removed]' else x)

df['selftext'].isnull().sum()
```




    8480




```python
# Drop all entries in 'selftext' with a null value.

df.dropna(subset=['selftext'], inplace=True)
```



## Initial Look at Data

### Null Values


```python
nulls = df.isnull().sum().sort_values(ascending=False)
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
      <td>adserver_click_url</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>1</th>
      <td>promoted_by</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>2</th>
      <td>promoted_url</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>3</th>
      <td>promoted</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>4</th>
      <td>original_link</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>5</th>
      <td>secure_media</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>6</th>
      <td>disable_comments</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>7</th>
      <td>href_url</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>8</th>
      <td>mobile_ad_url</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>9</th>
      <td>imp_pixel</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>10</th>
      <td>third_party_tracking</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>11</th>
      <td>third_party_tracking_2</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>12</th>
      <td>media</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>13</th>
      <td>adserver_imp_pixel</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>14</th>
      <td>promoted_display_name</td>
      <td>18208</td>
    </tr>
    <tr>
      <th>15</th>
      <td>distinguished</td>
      <td>18127</td>
    </tr>
    <tr>
      <th>16</th>
      <td>author_flair_text</td>
      <td>17884</td>
    </tr>
    <tr>
      <th>17</th>
      <td>author_flair_css_class</td>
      <td>17797</td>
    </tr>
    <tr>
      <th>18</th>
      <td>link_flair_css_class</td>
      <td>15842</td>
    </tr>
    <tr>
      <th>19</th>
      <td>link_flair_text</td>
      <td>15841</td>
    </tr>
    <tr>
      <th>20</th>
      <td>preview</td>
      <td>14695</td>
    </tr>
    <tr>
      <th>21</th>
      <td>post_hint</td>
      <td>14695</td>
    </tr>
  </tbody>
</table>
</div>



Right away we can see there are many columns with all null values. We can drop these right away as they are literally giving us no information.


```python
null_cols = []
for col in df:
    if df[col].isnull().sum() == df[col].isnull().count():
        null_cols.append(col)
    else:
        null_cols = null_cols

df = df.drop(null_cols, axis=1)
```

Now we are left with only columns with either 0 null values, or more than
14,000 null values. To see if these columns with thousands of nulls have
anything of note contained in them, I'm going to quickly look through the most common values of of each of them.


```python
lots_of_nulls = []
for col in df:
    if df[col].isnull().sum() > 14000:
        lots_of_nulls.append(col)
    else:
        lots_of_nulls = lots_of_nulls
        
for col in lots_of_nulls:
    print(df[col].value_counts()[:10], \n)
```

    py32bg              81
    commercial-indie    45
    py27bg              41
    py32bggh            29
    hobbyist            27
    none                25
    py3int              25
    noob                20
    intermediate        15
    py3intgh            13
    Name: author_flair_css_class, dtype: int64

    @LucklessSeven      17
    ☺                   10
    Nooblet Brewer      10
    HH0718               8
    Swift                7
    @TheThroneOfLies     7
    Trusted              6
    re.tar               6
    0 0                  6
    I'm horrible.        5
    Name: author_flair_text, dtype: int64

    moderator    81
    Name: distinguished, dtype: int64

    issue-resolved         894
    question               463
    help                   323
    solved                 182
    resolved                70
    discussion cat-talk     68
    mec                     49
    gen                     34
    unsolved                33
    discussion              30
    Name: link_flair_css_class, dtype: int64

    Solved          1054
    Question         463
    help             306
    Discussion       101
    Resolved          70
    [MECHANICAL]      49
    [GENERAL]         34
    Unsolved          33
    AdventOfCode      25
    solved!           23
    Name: link_flair_text, dtype: int64

    self    3513
    Name: post_hint, dtype: int64



The only thing of note I can see in these columns is that the 'distinguished' column is only used to designate if a moderator of that subreddit is making the post. There's also the 'link_flair_css_class' and 'link_flair_text' columns which are mostly used to categorize the type of post, but as they both have so many null values I'm assuming that only specific subreddits use the flairs. Because of this, the only use from all of these columns with more than 14,000 null values will be to make a column designating whether a post was made by a moderator or not.


```python
df['moderator'] = df['distinguished'].apply(lambda x: 1 if x == \
                                            'moderator' else 0)
```

I'm now going to drop all of the columns with more than 14,000 null values as I do not believe there is anymore useful information contained within them. We are now left with a dataset with no more null values.


```python
df = df.drop(lots_of_nulls, axis=1)
```


```python
boolean_columns = []
for col in df:
    if df[col].dtypes == 'bool':
        boolean_columns.append(col)
    else:
        boolean_columns = boolean_columns
```


```python
# Taking a look into the values in the boolean columns to see if 
# there is anything useful to be gleaned from them.

for col in boolean_columns:
    print(df[col].value_counts(), '\n')
```

    False    18208
    Name: archived, dtype: int64 
    
    False    18203
    True         5
    Name: contest_mode, dtype: int64 
    
    False    18208
    Name: hide_score, dtype: int64 
    
    True    18208
    Name: is_self, dtype: int64 
    
    False    18206
    True         2
    Name: locked, dtype: int64 
    
    False    18192
    True        16
    Name: over_18, dtype: int64 
    
    False    18208
    Name: quarantine, dtype: int64 
    
    False    18208
    Name: saved, dtype: int64 
    
    False    18208
    Name: spoiler, dtype: int64 
    
    False    18207
    True         1
    Name: stickied, dtype: int64 
    


It seems like all of the boolean columns are almost completely False and several are actually completely False. The only boolean column with any seemingly meaningful amount of True values is 'is_self', though because this dataset is made up only of self-posts, this is also a useless column as it would be on every entry if it was rightfully applied to all self-posts. Because of these reasons we are going to drop all of these columns.


```python
df = df.drop(boolean_columns, axis=1)
```

Taking a bit of a closer look into what our data contains now that the number of columns has been cut down quite significantly, we can see that there are plenty more columns that will provide nothing to our model. These include:
- **Author:** The Reddit username of whoever submitted the post.
- **Created UTC:** The unix timecode of when the post was submitted. As we are only trying to predict which subreddit the post was made to and not the score of the post (where time created would actually be relevant) I do not believe that this information would be at all useful.
- **Domain:** We have already used this column to distinguish whether or not an entry was actually a self-post or just a misclassified deleted link-post. This no longer contains any relevant information that is not contained within the 'subreddit' column.
- **Downs:** The number of downvotes that the post has received. This is a feature of Reddit that is no longer visible to users but has clearly not been completely phased out yet. Every entry in this column has a value of 0.0.
- **Edited:** Whether or not the post was edited after the original submission. Will have a value of False if it has not been edited, or a unix timestamp of when the post was edited if it has been.
- **Gilded:** How many Reddit golds that the post has received. Since it costs around 5$ to buy reddit gold and is used basically just as a glorified upvote, there are only 10 non-zero values in this column. For that reason I do not think it will provide anything to the model.
- **ID:** A unique post id given by Reddit.
- **Media Embed:** I believe this is where any relevant link information would go if we were doing looking at posts that aren't self-posts. Because we are not, all of the rows in this column are just empty dictionary objects.
- **Name:** I believe this is just another unique ID given to the post by reddit. Not sure why there are two different unique keys given (name and id) but I'd assume there is a reason.
- **Permalink:** The full reddit link to the specific post. Relevant information in here is already contained in other columns, such as subreddit name and post title.
- **Retrieved On:** The date and time that Richard would have retrieved the post on.
- **Secure Media Embed:** Seems to be the exact same as Media Embed as every entry is again an empty dictionary object.
- **Subreddit ID:** The unique ID given to each subreddit.
- **Thumbnail:** String values of either 'self' or'nsfw'. ** *From this I'll just make a 'nsfw' column with a value of 1 for entries with the thumbnail value of 'nsfw' and 0 for all others.* **
- **Ups:** The number of upvotes that the post has received. Because the downvotes are not visible, but the score is, in every entry 'score' is equal to 'ups'. Because of this we will remove 'ups' to avoid double-dipping on data.
- **URL:** The full url of the post. All relevant information is already contained in other columns.

After creating the one column 'nsfw' from all of these otherwise irrelevant columns, we will drop them all from the dataframe.


```python
# Make the 'nsfw' column as outlined above.

df['nsfw'] = df['thumbnail'].apply(lambda x: 1 if x == \
                                   'nsfw' else 0)
```


```python
# Drop all columns with irrelevant or duplicate information as
# outlined above.

df = df.drop(['author', 'created_utc', 'domain', 'downs', \
              'edited', 'gilded', 'id', 'media_embed', 'name', \
              'permalink', 'retrieved_on', 'secure_media_embed', \
              'subreddit_id', 'thumbnail', 'ups', 'url'], axis=1)
```


```python
# Creating a column called 'selftext_length' with the length of 
# the text of the post.

df['selftext_length'] = df['selftext'].apply(lambda x: len(x))
```


```python
# Creating a masked heatmap of the correlation between the three
# seemingly important numerical features that we have.

corr = df[['num_comments', 'score', 'selftext_length']].corr()

fig, ax = plt.subplots(figsize=(6,6))

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(corr, mask=mask, ax=ax, annot=True)

ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
```


![png](/images/2017-11-28-predicting-subreddit_files/2017-11-28-predicting-subreddit_30_0.png)


As we can see, the number of comments and the score are very highly correlated. While the length of the selftext is not at all correlated with either of them. I will keep both the score and the number of comments in the model for now, as I'm assuming that because most of my processing will be done with NLP I will end up with thousands of predictor columns anyways and have to use a model that is very robust to overfitting. I am happy with any additional variance a datapoint may be able to add.


Our dataset is now completely clean and contains only the columns that I believe will be relevant to predicting the subreddit of the post. We can now move onto the Natural Language Processing step in order to change the 'selftext' and 'title' columns into predictors that are actually usable for modeling.

## Natural Language Processing

Before we start onto changing our text data into data that is useful for modeling, we will have to split up our data into a training set and a test set. The values we are trying to predict are the subreddits, which we will need to change into numerical values, and our predictors will be everything else.


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, \
    TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
```


```python
# Creates a dictionary of all the subreddits in the dataset with 
# an integer attached that will correspond to only that subreddit.

subreddit_dictionary = {}
for ind, val in enumerate (df['subreddit'].value_counts()\
                                          .index.tolist()):
    subreddit_dictionary[val] = ind
```


```python
# Making our target column of integers attached to subreddits by 
# referencing the dictionary created above.

y = df['subreddit'].apply(lambda x: subreddit_dictionary[x])
```


```python
# Making our predictor columns to be the cleaned dataset minus just 
# the target column we have already extracted to y.

df_notarget = df.drop('subreddit', axis=1)
```


```python
X_train, X_test, y_train, y_test = train_test_split\
                (df_notarget, y, test_size=0.2, random_state=2017)
```

We now have our separate training and testing datasets. From here on out, everything we do to the training set to get it ready for modeling will also need to be applied to the testing set. We can do this in one fell swoop at the end of our preprocessing stage, before we get into the modeling.

Before we start into the Natural Language Processing we will need to extract the text columns ('selftext' and 'title') from both X_train and X_test into their own separate series. This is so that they can be processed individually by vectorizers before we add their output back to the rest of our data for modeling.


```python
X_train_selftext = X_train['selftext']
X_train_title = X_train['title']
X_test_selftext = X_test['selftext']
X_test_title = X_test['title']
```



```python
print(X_train_selftext)

# From looking through a bit of these 'selftext' values, we can see 
# that line breaks in the text are signified with '\n'. We will 
# replace these with a space for both the training and testing set.

X_train_selftext = X_train_selftext.apply(lambda x: x.replace('\n', ' '))
X_test_selftext = X_test_selftext.apply(lambda x: x.replace('\n', ' '))
```

    20917    My dad has a workstation PC as well as an inte...
    21670    Hello. I just reinstalled my windows 10 instal...
    16748    I recently order a gtx 1060 (gigabyte g1 gamin...
    5899     Hi! So I'm gonna be honest and make it maybe r...
    22327    Should I place them all in a Protocols.swift f...
    983      The problem is that I don't have a working dri...
    4455     turn on computer to find my bios are trying to...
    16743    Not homework just a personal project  \nso I'v...
    23765    I built a website that uses bootstrap-tables. ...
    25579    I've had a Samsung 840 Evo 250GB plugged into ...
    25739    Building a scraper that would run 5 days a wee...
    20242    I am making an app where a user can add a bunc...
    19933    Sorry if the title is ambiguous. I wasn't sure...
    7477     The course that I TA for has a new professor s...
    1362     I want to learn to code a game using visual st...
    936      To all the developers here: I would love to he...
    13818    http://imgur.com/a/Ysv1K\n\nI dont know why th...
    13589    Ok folks hear me out on this, I have bulk stan...
    4228     I used my Turtle Beach PX3 headset as a record...
    890      Firstly I appreciate anyone who takes the time...
    2994     I recently bought a GTX 1070 (MSI) and I've be...
    25531    I am creating a flask web app that is based on...
    13771    In case you want to help out with this thread,...
    14932    So to make it short I wanted to change motherb...
    4693     Few pages in my app have certain animations in...
    8587     This Dell laptop beeps in a series of 3 times ...
    4610     https://www.youtube.com/watch?v=LvUHrCj-beY  \...
    15406    I have a genuine copy of windows just to point...
    14635    I've put a background image and changed the op...
    3734     I built a new computer and have been having so...
                                   ...                        


We will be creating a text preprocesser in order to best turn our text data into something that is usable for modeling. We will be using the PorterStemmer to combine all derivations of root words into just the root, and stopwords to remove meaningless words from the text.


```python
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
```


```python
# Creating our text preprocesser. The cleaner function will remove 
# punctuation and numbers from the text, change it all into lowercase,
# and add all words that are not in the english stop-words list to a 
# list of final words, while stemming words with the same roots 
# together.

def cleaner(text):
    stemmer = PorterStemmer()
    stop = stopwords.words('english')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    text = text.lower().strip()
    final_text = []
    for w in text.split():
        if w not in stop:
            final_text.append(stemmer.stem(w.strip()))
    return ' '.join(final_text)
```

Because our text_preprocesser we just created strips all punctuation and digits from the text we run it through, I will manually create binary classification columns for where the title and text of a post contains punctuation or digits of some sort. Utilizing some simple regular expressions to account for possible differences in capitalization and spacing when needed.


```python
# title
X_train['title_contains_?'] = \
    X_train_title.str.contains('\?').astype(int)
X_train['title_contains_['] = \
    X_train_title.str.contains('\[').astype(int)
X_train['title_contains_C++'] = \
    X_train_title.str.contains('[Cc]\+\+').astype(int)
X_train['title_contains_Windows7or8'] = \
    X_train_title.str.contains('[Ww]indows [78]').astype(int)
X_train['title_contains_Windows10'] = \
    X_train_title.str.contains('[Ww]indows 10').astype(int)
X_train['title_contains_64bit'] = \
    X_train_title.str.contains('64[\s\S][Bb]it').astype(int)
X_train['title_contains_32bit'] = \
    X_train_title.str.contains('32[\s\S][Bb]it').astype(int)
X_train['title_contains_64bit'] = \
    X_train_title.str.contains('64[\s\S][Bb]it').astype(int)
X_train['title_contains_()'] = \
    X_train_title.str.contains('\(\)').astype(int)
X_train['title_contains_underscore'] = \
    X_train_title.str.contains('\_').astype(int)

# selftext
X_train['selftext_contains_?'] = \
    X_train_selftext.str.contains('\?').astype(int)
X_train['selftext_contains_['] = \
    X_train_selftext.str.contains('\[').astype(int)
X_train['selftext_contains_C++'] = \
    X_train_selftext.str.contains('[Cc]\+\+').astype(int)
X_train['selftext_contains_Windows7or8'] = \
    X_train_selftext.str.contains('[Ww]indows [78]').astype(int)
X_train['selftext_contains_Windows10'] = \
    X_train_selftext.str.contains('[Ww]indows 10').astype(int)
X_train['selftext_contains_64bit'] = \
    X_train_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_train['selftext_contains_32bit'] = \
    X_train_selftext.str.contains('32[\s\S][Bb]it').astype(int)
X_train['selftext_contains_64bit'] = \
    X_train_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_train['selftext_contains_()'] = \
    X_train_selftext.str.contains('\(\)').astype(int)
X_train['selftext_contains_underscore'] = \
    X_train_selftext.str.contains('\_').astype(int)
```



```python
# same columns as above for the test data

X_test['title_contains_?'] = \
    X_test_title.str.contains('\?').astype(int)
X_test['title_contains_['] = \
    X_test_title.str.contains('\[').astype(int)
X_test['title_contains_C++'] = \
    X_test_title.str.contains('[Cc]\+\+').astype(int)
X_test['title_contains_Windows7or8'] = \
    X_test_title.str.contains('[Ww]indows [78]').astype(int)
X_test['title_contains_Windows10'] = \
    X_test_title.str.contains('[Ww]indows 10').astype(int)
X_test['title_contains_64bit'] = \
    X_test_title.str.contains('64[\s\S][Bb]it').astype(int)
X_test['title_contains_32bit'] = \
    X_test_title.str.contains('32[\s\S][Bb]it').astype(int)
X_test['title_contains_64bit'] = \
    X_test_title.str.contains('64[\s\S][Bb]it').astype(int)
X_test['title_contains_()'] = \
    X_test_title.str.contains('\(\)').astype(int)
X_test['title_contains_underscore'] = \
    X_test_title.str.contains('_').astype(int)

X_test['selftext_contains_?'] = \
    X_test_selftext.str.contains('\?').astype(int)
X_test['selftext_contains_['] = \
    X_test_selftext.str.contains('\[').astype(int)
X_test['selftext_contains_C++'] = \
    X_test_selftext.str.contains('[Cc]\+\+').astype(int)
X_test['selftext_contains_Windows7or8'] = \
    X_test_selftext.str.contains('[Ww]indows [78]').astype(int)
X_test['selftext_contains_Windows10'] = \
    X_test_selftext.str.contains('[Ww]indows 10').astype(int)
X_test['selftext_contains_64bit'] = \
    X_test_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_test['selftext_contains_32bit'] = \
    X_test_selftext.str.contains('32[\s\S][Bb]it').astype(int)
X_test['selftext_contains_64bit'] = \
    X_test_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_test['selftext_contains_()'] = \
    X_test_selftext.str.contains('\(\)').astype(int)
X_test['selftext_contains_underscore'] = \
    X_test_selftext.str.contains('\_').astype(int)
```


### Count Vectorizer

Use CountVectorizer and our text-preprocesser from above to process our selftext data. I will also limit the words chosen to only those that occur in a minimum of 1/1000th of the entries, and a maximum of half of the entries.


```python
cv = CountVectorizer(preprocessor=cleaner, min_df=0.001, max_df=0.50)
train_selftext_cv = cv.fit(X_train_selftext)

# Because it's almost a certainty that there will be shared common 
# words between the selftext and the title of a post, I will need 
# to create feature names with the prefix 'selftext_' to make sure 
# that I have separate entries.

selftext_feature_names = []
for feature in cv.get_feature_names():
    selftext_feature_names.append('selftext_' + feature)

# Create a dataframe from the CountVectorizer processed data using 
# the feature names from above

train_selftext_cv_df = pd.DataFrame(cv.transform(X_train_selftext)\
                       .todense(), columns=selftext_feature_names)

train_selftext_cv_df.shape
```




    (14566, 3323)




```python
# Now do the same as above for our testing selftext data.

test_selftext_cv_df = pd.DataFrame(cv.transform(X_test_selftext)\
                      .todense(), columns=selftext_feature_names)
test_selftext_cv_df.shape
```




    (3642, 3323)



Repeat the same processes from the above two steps for our title data, making sure to add the prefix of 'title_' to our features produced here. I am printing out the shapes of our two new dataframes just to confirm that everything has went as planned.



```python
cv = CountVectorizer(preprocessor=cleaner, min_df=0.001, max_df=0.50)
train_title_cv = cv.fit(X_train_title)

title_feature_names = []
for feature in cv.get_feature_names():
    title_feature_names.append('title_' + feature)

train_title_cv_df = pd.DataFrame(cv.transform(X_train_title)\
                    .todense(), columns=title_feature_names)

test_title_cv_df = pd.DataFrame(cv.transform(X_test_title)\
                   .todense(), columns=title_feature_names)

print(train_title_cv_df.shape, '\n', test_title_cv_df.shape)
```

    (14566, 981) 
     (3642, 981)


#### TFIDF Vectorizer

Using pretty much the same process as we used for Count Vectorizer, we will also process our text with Tfidf Vectorizer without any limits on the words depending on how common they are, and cut down the number of features output by using Truncated SVD, a method of PCA best used on the sparse matrices output through NLP.


```python
from sklearn.decomposition import TruncatedSVD
```

#### Selftext


```python
tfidf = TfidfVectorizer(preprocessor=cleaner)
train_selftext_tfidf = tfidf.fit(X_train_selftext)

selftext_feature_names = []
for feature in tfidf.get_feature_names():
    selftext_feature_names.append('selftext_' + feature)

train_selftext_tfidf_df = pd.DataFrame(tfidf.transform\
    (X_train_selftext).todense(), columns=selftext_feature_names)

test_selftext_tfidf_df = pd.DataFrame(tfidf.transform\
    (X_test_selftext).todense(), columns=selftext_feature_names)

print(train_selftext_tfidf_df.shape, '\n', \
      test_selftext_tfidf_df.shape)
```

    (14566, 49629) 
     (3642, 49629)


Tfidf Vectorizer created 49629 columns from our selftext text, we'll try cutting that down to 10000 using Truncated SVD, and see how much of the variance in the data is explained with only these heavily reduced amount of columns.


```python
selftext_tsvd = TruncatedSVD(n_components=10000)
selftext_tsvd.fit(train_selftext_tfidf_df.values)
```




    TruncatedSVD(algorithm='randomized', n_components=10000, n_iter=5,
           random_state=None, tol=0.0)




```python
plt.plot(range(10000), selftext_tsvd.explained_variance_ratio_\
                                                     .cumsum())

plt.title('Selftext TSVD', fontsize=20)
plt.xlabel('Number of Components in Truncated SVD', fontsize=14)
plt.ylabel('Percentage of Expained Variance', fontsize=14)

# About 99% of the variance in the selftext data is explained with 
# only 10000 of the original 49629 columns, this should save A LOT 
# of time while modeling and not have too much of an impact on our 
# model
```




    <matplotlib.text.Text at 0x114192b70>




![png](/images/2017-11-28-predicting-subreddit_files/2017-11-28-predicting-subreddit_66_1.png)



```python
X_train_selftext_tfidf = selftext_tsvd.transform\
    (train_selftext_tfidf_df.values)
X_test_selftext_tfidf = selftext_tsvd.transform\
    (test_selftext_tfidf_df.values)
```


```python
# Creating dataframes from our transformed selftext data, making sure 
# to give them a list of column names. I will give the transformed 
# title data column names continuing on from 10000 so that when I go 
# to merge the dataframes all back together there is no overlap. I 
# am also setting the index to the index of X_train as the NLP 
# transformations have reset them back to ascending from 0 as 
# opposed to the specific rows that train_test_split chose for 
# X_train.

X_train_selftext_tfidf_df = pd.DataFrame(X_train_selftext_tfidf, 
        columns=list(range(0, 10000)), index=X_train.index)
X_test_selftext_tfidf_df = pd.DataFrame(X_test_selftext_tfidf,
        columns=list(range(0, 10000)), index=X_test.index)
```

#### Title


```python
tfidf = TfidfVectorizer(preprocessor=cleaner)
train_title_tfidf = tfidf.fit(X_train_title)

title_feature_names = []
for feature in tfidf.get_feature_names():
    title_feature_names.append('title_' + feature)

train_title_tfidf_df = pd.DataFrame(tfidf.transform(X_train_title)\
    .todense(), columns=title_feature_names)

test_title_tfidf_df = pd.DataFrame(tfidf.transform(X_test_title)\
    .todense(), columns=title_feature_names)

print(train_title_tfidf_df.shape, '\n', test_title_tfidf_df.shape)
```

    (14566, 8723) 
     (3642, 8723)


Tfidf Vectorizer created 8723 columns from our title text, we'll try cutting that down to 1500 using Truncated SVD.


```python
title_tsvd = TruncatedSVD(n_components=1500)
title_tsvd.fit(train_title_tfidf_df.values)
```




    TruncatedSVD(algorithm='randomized', n_components=1500, n_iter=5,
           random_state=None, tol=0.0)




```python
plt.plot(range(1500), title_tsvd.explained_variance_ratio_.cumsum())

plt.title('Title TSVD', fontsize=20)
plt.xlabel('Number of Components in Truncated SVD', fontsize=14)
plt.ylabel('Percentage of Expained Variance', fontsize=14)

# About 78% of the variance in the title data is explained with only 
# 1500 of the original 8723 columns.
```




    <matplotlib.text.Text at 0x1a1ca42470>




![png](/images/2017-11-28-predicting-subreddit_files/2017-11-28-predicting-subreddit_73_1.png)



```python
X_train_title_tfidf = title_tsvd.transform\
    (train_title_tfidf_df.values)
X_test_title_tfidf = title_tsvd.transform\
    (test_title_tfidf_df.values)
```


```python
X_train_title_tfidf_df = pd.DataFrame(X_train_title_tfidf,
    columns=list(range(10000,11501)), index=X_train.index)
X_test_title_tfidf_df = pd.DataFrame(X_test_title_tfidf,
    columns=list(range(10000,11501)), index=X_test.index)
```

## Getting Ready for Modeling

Now that we have processed all our our text data and have it in 4 separate dataframes (selftext and title for both the training and the testing set) we will need to put all of our processed data back together into just one dataframe each for both X_train and X_test. However, since we have transformed our data with NLP using both CountVectorizer and TfidfVectorizer, I will create an X_train and X_test dataframe for each of the two methods.

However, before we can do that we will want to scale our numerical data that we have not done anything with yet.


```python
# We can safely drop our text columns from both X_train and X_test 
# as they have already been extracted and processed.

X_train = X_train.drop(['selftext', 'title'], axis=1)
X_test = X_test.drop(['selftext', 'title'], axis=1)
```


We have 23 numerical columns that we want to use in our model. Because 20 of them are binary classification columns, we will not need to scale those, but the other three will definitely benefit from scaling.


```python
X_train[['num_comments', 'score', 'selftext_length']].describe()
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
      <th>num_comments</th>
      <th>score</th>
      <th>selftext_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>14566.000000</td>
      <td>14566.000000</td>
      <td>14566.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7.041398</td>
      <td>8.857270</td>
      <td>765.162433</td>
    </tr>
    <tr>
      <th>std</th>
      <td>57.811907</td>
      <td>232.363592</td>
      <td>873.089576</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>321.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>547.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>930.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6109.000000</td>
      <td>26573.000000</td>
      <td>31185.000000</td>
    </tr>
  </tbody>
</table>
</div>



From looking at the description of our 3 non-binary-classification columns, we can see that there some extreme outliers in all 3. We will want to use a scaler that is very robust to outliers, and for that reason we will be using QuantileTransformer to completely negate the outliers in the data.


```python
from sklearn.preprocessing import RobustScaler, QuantileTransformer
```


```python
# Here we are scaling our three columns using QuantileTransformer 
# and replacing the original values in X_train with the scaled 
# values.

qt = QuantileTransformer()
X_train_qt = qt.fit_transform(X_train[['num_comments', \
                              'score', 'selftext_length']])

X_train[['num_comments', 'score', 'selftext_length']] = X_train_qt

X_train[['num_comments', 'score', 'selftext_length']].describe()
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
      <th>num_comments</th>
      <th>score</th>
      <th>selftext_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.456600e+04</td>
      <td>1.456600e+04</td>
      <td>1.456600e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.857194e-01</td>
      <td>4.940484e-01</td>
      <td>5.000060e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.088103e-01</td>
      <td>2.756758e-01</td>
      <td>2.886994e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000e-07</td>
      <td>1.000000e-07</td>
      <td>1.000000e-07</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.267267e-01</td>
      <td>3.718719e-01</td>
      <td>2.507508e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.734735e-01</td>
      <td>3.718719e-01</td>
      <td>4.994995e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.662663e-01</td>
      <td>7.012012e-01</td>
      <td>7.497497e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.999999e-01</td>
      <td>9.999999e-01</td>
      <td>9.999999e-01</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Performing the same scaling as above to the relevant testing 
# data columns.

X_test_qt = qt.transform(X_test[['num_comments', 'score', \
                                 'selftext_length']])

X_test[['num_comments', 'score', 'selftext_length']] = X_test_qt
```



```python
# Just confirming that all of our separate dataframes have the 
# same amount of rows, and the matching dataframes between the 
# training and testing sets have the same amount of columns so 
# that we don't run into issues when we try to add them together 
# and use them for our model.

print('Count Vectorizer:')
print('Training Data:', X_train.shape, train_selftext_cv_df.shape, \
      train_title_cv_df.shape)
print('Testing Data:', X_test.shape, test_selftext_cv_df.shape, \
      test_title_cv_df.shape)
print('\n')
print('Tfidf Vectorizer with Truncated SVD:')
print('Training Data:', X_train.shape, \
      X_train_selftext_tfidf_df.shape, X_train_title_tfidf_df.shape)
print('Testing Data:', X_test.shape, \
      X_test_selftext_tfidf_df.shape, X_test_title_tfidf_df.shape)
```

    Count Vectorizer:
    Training Data: (14566, 23) (14566, 3323) (14566, 981)
    Testing Data: (3642, 23) (3642, 3323) (3642, 981)
    
    
    Tfidf Vectorizer with Truncated SVD:
    Training Data: (14566, 23) (14566, 10000) (14566, 1500)
    Testing Data: (3642, 23) (3642, 10000) (3642, 1500)



```python
# Merging our three dataframes for our training data back into one 
# dataframe that we will use for modeling. Using a left merge on the 
# last column of each of the original dataframes so that we do not 
# drop any columns.

modeling_cv_X_train = pd.merge(X_train, train_selftext_cv_df, 
    left_on=X_train.columns[-1], right_index=True, 
    how='left', sort=False)
modeling_cv_X_train = pd.merge(modeling_cv_X_train, train_title_cv_df,
    left_on=modeling_cv_X_train.columns[-1], right_index=True, 
    how='left', sort=False)
```


```python
# Now doing the same merging of dataframes for our testing data.

modeling_cv_X_test = pd.merge(X_test, test_selftext_cv_df, 
    left_on=X_test.columns[-1], right_index=True, 
    how='left', sort=False)
modeling_cv_X_test = pd.merge(modeling_cv_X_test, test_title_cv_df, 
    left_on=modeling_cv_X_test.columns[-1], right_index=True, 
    how='left', sort=False)

print(modeling_cv_X_train.shape, '\n', modeling_cv_X_test.shape)
```

    (14566, 4327) 
     (3642, 4327)



```python
# Now doing tfidf data.

modeling_tfidf_X_train = pd.concat([X_train, X_train_title_tfidf_df], 
                                  axis=1)
modeling_tfidf_X_train = pd.concat([modeling_tfidf_X_train, 
                                  X_train_selftext_tfidf_df], axis=1)

modeling_tfidf_X_test = pd.concat([X_test, X_test_title_tfidf_df], 
                                  axis=1)
modeling_tfidf_X_test = pd.concat([modeling_tfidf_X_test, 
                                  X_test_selftext_tfidf_df], axis=1)
```


```python
print(modeling_tfidf_X_train.shape, modeling_tfidf_X_test.shape)
```

    (14566, 11523) (3642, 11523)


# Modeling


```python
fig, ax = plt.subplots(figsize=(20,15))

plt.bar(np.arange(len(df.subreddit.value_counts())), 
    df.subreddit.value_counts(), tick_label=\
    df.subreddit.value_counts().values)

plt.xticks(np.arange(len(df.subreddit.value_counts())), 
    df['subreddit'].value_counts().index, rotation=90, fontsize=22)

rects = ax.patches

labels = df.subreddit.value_counts()

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, 
            ha='center', va='bottom', fontsize=15)

plt.yticks(fontsize=20)

plt.xlabel('Subreddits', fontsize=40)
plt.ylabel('Number of Occurences in Dataset', fontsize=40)
plt.title('Number of Each Subreddit in Dataset', fontsize=50)

plt.show()
```


![png](/images/2017-11-28-predicting-subreddit_files/2017-11-28-predicting-subreddit_94_0.png)


I know it is generally bad practice to use pie charts, but I just wanted to show the rough distribution of subreddits in the dataset. I believe that the piechart gets this across very well at a quick glance.


```python
pie_df = pd.DataFrame([df.subreddit.value_counts()[0], 
    df.subreddit.value_counts()[1],
    df.subreddit.value_counts()[2],
    df.subreddit.value_counts()[3:].sum()], 
    index=[df['subreddit'].value_counts().index[0],
    df['subreddit'].value_counts().index[1],
    df['subreddit'].value_counts().index[2], 'bottom ' + \
        str(len(df.subreddit.value_counts()[3:]))+' subreddits'])

pie_df.plot(kind='pie', subplots=True, figsize=(12,12), fontsize=26,
    legend=False, title='Subreddit Distribution in Dataset')

# Cannot seem to be able to figure out how to increase the size of 
# the title in pandas plots
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x1a43995e10>], dtype=object)




![png](/images/2017-11-28-predicting-subreddit_files/2017-11-28-predicting-subreddit_96_1.png)


As stated before, we will be trying to predict the subreddit of each post in our dataset. As we can see in the graphs above, the vast majority of entries in the dataset belong to just a few different subreddits. With techsupport having a little more than half, and a little bit less than a quarter being shared between learnprogramming and learnpython, that leaves only about a quarter of all entries split between the remaining 32 subreddits.

This could cause issues with our model, as I believe it will be very unlikely to predict any of the subreddits that don't even have a visible bar on the barchart. There are just too few entries to train on. I also believe that this will be balanced out with our model predicting much too many techsupport values, and also probably learnprogramming and learnpython. 


```python
# Calculating the baseline accuracy of our model - the percentage 
# of correct classifications that we want to beat, in order for our 
# model to be better than just guessing that every entry's subreddit 
# is techsupport.

baseline = df['subreddit'].value_counts()[0] / \
    df['subreddit'].value_counts().sum()

print(baseline)
```

    0.525538224956


Our baseline accuracy for our model is about 52.6%.


```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
```

### Using Count Vectorizer Data


```python
randomforest = RandomForestClassifier(n_jobs=-1)
randomforest.fit(modeling_cv_X_train, y_train)

randomforest.score(modeling_cv_X_train, y_train)
```




    0.95825895922010162




```python
cross_val_score(randomforest, modeling_cv_X_test, y_test, 
                cv=5, n_jobs=-1).mean()
```


    0.52304268922533714




```python
gbc = GradientBoostingClassifier()

gbc.fit(modeling_cv_X_train, y_train)
gbc.score(modeling_cv_X_train, y_train)
```




    0.67067142660991352




```python
cross_val_score(gbc, modeling_cv_X_test, y_test,
               cv=5, n_jobs=-1).mean()
```


    0.57992537278572309




```python
lr = LogisticRegression(n_jobs=-1)

lr.fit(modeling_cv_X_train, y_train)
lr.score(modeling_cv_X_train, y_train)
```




    0.5968007689139091




```python
cross_val_score(lr, modeling_cv_X_test, y_test, cv=5).mean()
```





    0.58958924549102965



Our CountVectorizer data did not perform great. The base Gradient Boosting Classification and Logistic Regression models was able to beat the baseline, but only by ~2-3%. I will now try the TfidfVectorizer with Truncated SVD data to see if it performs better.

### Using Tfidf Vectorizer Truncated SVD Data


```python
lr = LogisticRegression()
lr.fit(modeling_tfidf_X_train, y_train)

lr.score(modeling_tfidf_X_train, y_train)
```




    0.87127557325278049




```python
cross_val_score(lr, modeling_tfidf_X_test, y_test, cv=5).mean()
```





    0.719468068916014



Right off the bat the Tfidf with TSVD data performs much better, with a 72% testing set accuracy as opposed to the 59% of the best model using the Count Vectorizer data. However, our model does seem to be much more overfit now, with a 15% higher accuracy on the training set than the test set compared to less than a 1% difference with the Count Vectorizer data. Tfidf Vectorizer still gives us the most accurate results on the test set though so I will now attempt to tune this model and see how high we can get it.




```python
# This gridsearch has been narrowed down over several iterations
# to be where it is below

params = {
    'C': [4, 5, 6, 7,],
    'multi_class': ['ovr', 'multinomial'],
    'solver': ['sag', 'newton-cg', 'lbfg2']
}

gs_lr = LogisticRegression()

clf = GridSearchCV(gs_lr, param_grid=params, n_jobs=-1)

clf.fit(modeling_tfidf_X_train, y_train)

print(clf.best_params_)
print(clf.best_score_)
```


    {'C': 5, 'multi_class': 'multinomial', 'solver': 'sag'}
    0.790608265825



```python
mn_lr_C5 = LogisticRegression(solver='sag', multi_class='multinomial',
                             C=5, n_jobs=-1)

mn_lr_C5.fit(modeling_tfidf_X_train, y_train)
mn_lr_C5.score(modeling_tfidf_X_train, y_train)
```


    0.99237951393656465


```python
cross_val_score(mn_lr_C5, modeling_tfidf_X_test, y_test, cv=5, n_jobs=-1).mean()
```



    0.74966167184815924



Our tuned model has an accuracy of ~75.2%, beating the baseline by ~22.6%. For a dataset with 35 different classifications, with many of those classifications having less than 100 entries and therefore very little to train on, I am quite happy with this result. Time to evaluate the model a little bit more in-depth and see where it performed well and where it did not. If it is as I expected, it will perform the best on techsupport, learnprogramming, and learnpython (labels 0, 1 and 2) as those are the most common classifications by far, and have many false positives for each. I would also assume that there are quite a few classifications that do not even get predicted at all as six of them have less than ten entries in the entire dataset.

# Evaluating our Model


```python
from sklearn.metrics import confusion_matrix, classification_report
```


```python
predicted = mn_lr_C5.predict(modeling_tfidf_X_test)
print(classification_report(y_test, predicted))
```

                 precision    recall  f1-score   support
    
              0       0.94      0.99      0.96      1891
              1       0.61      0.74      0.67       548
              2       0.60      0.74      0.66       305
              3       0.79      0.85      0.82       155
              4       0.62      0.57      0.60        84
              5       0.58      0.42      0.49        85
              6       0.52      0.33      0.41        81
              7       0.81      0.85      0.83        54
              8       0.35      0.10      0.15        61
              9       0.56      0.38      0.45        58
             10       0.79      0.41      0.54        46
             11       0.89      0.39      0.54        44
             12       0.35      0.31      0.33        29
             13       0.77      0.34      0.48        29
             14       0.69      0.35      0.46        26
             15       0.91      0.42      0.57        24
             16       0.20      0.05      0.08        19
             17       0.33      0.07      0.12        14
             18       0.43      0.27      0.33        11
             19       1.00      0.22      0.36         9
             20       0.67      0.50      0.57        12
             21       1.00      0.50      0.67         8
             22       0.00      0.00      0.00         7
             23       1.00      0.12      0.22         8
             24       0.00      0.00      0.00         6
             25       0.00      0.00      0.00         7
             26       0.67      0.25      0.36         8
             27       0.50      0.33      0.40         3
             28       0.00      0.00      0.00         5
             30       1.00      1.00      1.00         1
             31       0.00      0.00      0.00         3
             32       0.00      0.00      0.00         0
             33       1.00      1.00      1.00         1
    
    avg / total       0.79      0.80      0.78      3642
    



```python
# Giving the classification report the matching subreddit names 
# as labels.

predicted_classes = df.subreddit.value_counts()\
    .index[:30].append(df.subreddit.value_counts().index[31:34])

print(classification_report(y_test, predicted, 
                            target_names=predicted_classes))
```

                      precision    recall  f1-score   support
    
         techsupport       0.94      0.99      0.96      1891
    learnprogramming       0.61      0.74      0.67       548
         learnpython       0.60      0.74      0.66       305
             gamedev       0.79      0.85      0.82       155
          web_design       0.62      0.57      0.60        84
            javahelp       0.58      0.42      0.49        85
          javascript       0.52      0.33      0.41        81
             csshelp       0.81      0.85      0.83        54
              Python       0.35      0.10      0.15        61
      iOSProgramming       0.56      0.38      0.45        58
               linux       0.79      0.41      0.54        46
         engineering       0.89      0.39      0.54        44
               swift       0.35      0.31      0.33        29
     computerscience       0.77      0.34      0.48        29
              django       0.69      0.35      0.46        26
                 PHP       0.91      0.42      0.57        24
                 css       0.20      0.05      0.08        19
                java       0.33      0.07      0.12        14
                HTML       0.43      0.27      0.33        11
                ruby       1.00      0.22      0.36         9
               flask       0.67      0.50      0.57        12
             compsci       1.00      0.50      0.67         8
          technology       0.00      0.00      0.00         7
                 cpp       1.00      0.12      0.22         8
               html5       0.00      0.00      0.00         6
              pygame       0.00      0.00      0.00         7
              jquery       0.67      0.25      0.36         8
                perl       0.50      0.33      0.40         3
                lisp       0.00      0.00      0.00         5
          programmer       1.00      1.00      1.00         1
             IPython       0.00      0.00      0.00         3
    inventwithpython       0.00      0.00      0.00         0
              netsec       1.00      1.00      1.00         1
    
         avg / total       0.79      0.80      0.78      3642
    




```python
# The two classifications which did not appear in the test set or 
# get predicted by the model and are therefore omitted from the 
# classification report. Unfortunately, a confusion matrix for
# this 36 class classification problem is much to large to fit
# here while still being readable.

print(df.subreddit.value_counts().index[29], 
      df.subreddit.value_counts().index[34])
```

    programmer pystats



```python
# Just confirming that index 29 (programmer) and 34 (pystats) from 
# above did not appear in y_test. 32 (inventwithpython) also did not 
# appear in testing set, but had one predicted value with was actually
# in the subreddit learnwithpython

y_test.value_counts()
```




    0     1891
    1      548
    2      305
    3      155
    5       85
    4       84
    6       81
    8       61
    9       58
    7       54
    10      46
    11      44
    13      29
    12      29
    14      26
    15      24
    16      19
    17      14
    20      12
    18      11
    19       9
    26       8
    23       8
    21       8
    22       7
    25       7
    24       6
    28       5
    27       3
    31       3
    33       1
    30       1
    Name: subreddit, dtype: int64


So, as predicted, the multiple classifications with very few entries caused some minor problems with my model, though not only in the way I originally anticipated. There are three classifications (inventwithpython, programmer and pystats) that did not have a single entry in my testing set as there were too few values and the test_size of 20% did not happen to fall on any of them.

Also as predicted, my model did extremely well at predicting whether or not the post was in techsupport, but did noticeably worse for every other subreddit. Unfortunately, my model was not as good as I expected it would be for the other two sizable classes, learnprogramming and learnpython, but it did better than expected in the fourth most populous class gamedev.

The areas where my model is most inaccurate from a scan of the confusion matrix on my end are the classifications with fewer entries that also closely relate in topic to more populous classifications. For example: the model only correctly predicted 7 of the Python subreddit, but misclassified 33 of them as learnpython and 13 as learnprogramming, much more popular subreddits. css also only had one true positive, but three misclassifications in both techsupport and learnprogramming.

# Next Steps

Because I completely got rid of the deleted and removed posts before modeling in order to more accurately predict the subreddit of posts which actual 'selftext' values and rely less on the natural language processed 'title' columns, I would like to make a separate model for the entries I negated. It would be interesting to see how accurately I could predict the subreddit of a deleted or removed post using just the processed 'title' columns compared to the accuracy of the model with everything included.

I would also like to see if I could add an ROC curve to my model evaluation piece. I could not seem to get it to work at all for my multi-class classification as I had only done it before with binary classification problems. I'm assuming there is an easy fix to the problems I had but could not figure it out in my limited time I had left.
