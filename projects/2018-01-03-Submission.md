---
layout: page
title: Predicting Subreddits
date: 2018-01-03
published: true
---

# Project 4 - Reddit Post Analysis

This dataset contains all of the posts made to a specific set of programming-oriented subreddits in the month of December. We will make a model that attempts to accurately predict which subreddit the post was made to.

## Quick Fix to Dataset

The data has (supposedly) been subset to include only self-posts (posts where the the writer has submitted text instead of a link). 

However, when I was taking a deeper dive into the data, I noticed that not only self-posts were included within it. I also noticed that all of the non-self posts that had gotten through were deleted or removed from the subreddit. 

When a post is deleted (by the user) or removed (by a moderator/admin), it gives the text of the post, which would not have existed before removal, a value of [deleted] or [removed]. I am assuming that this is what caused it to slip by Richard's filter as the column 'selftext' was no longer empty. 

Since I will be relying mostly on Natural Language Processing for this model, and posts which are deleted or removed no longer have any selftext to process, I will actually be dropping all entries with a 'selftext' value of [deleted] or [removed]. This will solve both of these problems and make the model comparatively more reliant on the selftext and less reliant on the title (which remains even when a post is deleted or removed). To do this quick fix I did:


```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import timeit
```


```python
df = pd.read_csv('reddit_posts.csv')
```


```python
df['domain'].value_counts()
```




    self.techsupport                       11423
    self.learnprogramming                   3448
    self.learnpython                        1724
    self.gamedev                            1191
    self.technology                         1152
    self.web_design                          749
    self.Python                              566
    self.javahelp                            536
    self.javascript                          482
    self.linux                               465
    self.engineering                         454
    self.csshelp                             393
    self.iOSProgramming                      315
    self.swift                               249
    youtube.com                              188
    self.PHP                                 175
    self.computerscience                     159
    self.compsci                             147
    self.java                                138
    self.django                              124
    self.netsec                               97
    self.css                                  94
    self.HTML                                 85
    self.cpp                                  83
    github.com                                75
    self.ruby                                 72
    self.flask                                71
    youtu.be                                  63
    i.redd.it                                 58
    self.html5                                50
                                           ...  
    beconnected.in                             1
    raspberrypi.org                            1
    bharatbook.com                             1
    audero.it                                  1
    m.imgur.com                                1
    articles.abilogic.com                      1
    blogcodeclouds.tumblr.com                  1
    lootumaza.com                              1
    ianslive.in                                1
    blog.baasil.io                             1
    quizpush.com                               1
    whloss.com                                 1
    opensource.com                             1
    siliconbeat.com                            1
    feedass.com                                1
    hpsupporthelplinenumber.blogspot.in        1
    singleton11.github.io                      1
    all4webs.com                               1
    geek.com                                   1
    arstechnica.co.uk                          1
    gdeepak.com                                1
    arvrwatch.com                              1
    engineerlive.com                           1
    mobitabspecs.com                           1
    vikings.net                                1
    techfeeds360.com                           1
    aliexpress.com                             1
    lambda.bugyo.tk                            1
    lacomhps.com                               1
    npmjs.com                                  1
    Name: domain, Length: 871, dtype: int64




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
df['selftext'] = df['selftext'].apply(lambda x: np.nan if x == \
                                    '[deleted]' or x == '[removed]' else x)

df['selftext'].isnull().sum()
```




    8480




```python
# Drop all rows in 'selftext' with a null value (all deleted or removed
# posts).
df.dropna(subset=['selftext'], inplace=True)
df.shape
```




    (18208, 53)



## Initial Look at Data


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 18208 entries, 0 to 26686
    Data columns (total 53 columns):
    adserver_click_url        0 non-null float64
    adserver_imp_pixel        0 non-null float64
    archived                  18208 non-null bool
    author                    18208 non-null object
    author_flair_css_class    411 non-null object
    author_flair_text         324 non-null object
    contest_mode              18208 non-null bool
    created_utc               18208 non-null int64
    disable_comments          0 non-null float64
    distinguished             81 non-null object
    domain                    18208 non-null object
    downs                     18208 non-null float64
    edited                    18208 non-null object
    gilded                    18208 non-null float64
    hide_score                18208 non-null bool
    href_url                  0 non-null float64
    id                        18208 non-null object
    imp_pixel                 0 non-null float64
    is_self                   18208 non-null bool
    link_flair_css_class      2366 non-null object
    link_flair_text           2367 non-null object
    locked                    18208 non-null bool
    media                     0 non-null object
    media_embed               18208 non-null object
    mobile_ad_url             0 non-null float64
    name                      18208 non-null object
    num_comments              18208 non-null float64
    original_link             0 non-null float64
    over_18                   18208 non-null bool
    permalink                 18208 non-null object
    post_hint                 3513 non-null object
    preview                   3513 non-null object
    promoted                  0 non-null float64
    promoted_by               0 non-null float64
    promoted_display_name     0 non-null float64
    promoted_url              0 non-null float64
    quarantine                18208 non-null bool
    retrieved_on              18208 non-null float64
    saved                     18208 non-null bool
    score                     18208 non-null float64
    secure_media              0 non-null object
    secure_media_embed        18208 non-null object
    selftext                  18208 non-null object
    spoiler                   18208 non-null bool
    stickied                  18208 non-null bool
    subreddit                 18208 non-null object
    subreddit_id              18208 non-null object
    third_party_tracking      0 non-null float64
    third_party_tracking_2    0 non-null float64
    thumbnail                 18208 non-null object
    title                     18208 non-null object
    ups                       18208 non-null float64
    url                       18208 non-null object
    dtypes: bool(10), float64(19), int64(1), object(23)
    memory usage: 6.3+ MB



```python
df.isnull().sum()
```




    adserver_click_url        18208
    adserver_imp_pixel        18208
    archived                      0
    author                        0
    author_flair_css_class    17797
    author_flair_text         17884
    contest_mode                  0
    created_utc                   0
    disable_comments          18208
    distinguished             18127
    domain                        0
    downs                         0
    edited                        0
    gilded                        0
    hide_score                    0
    href_url                  18208
    id                            0
    imp_pixel                 18208
    is_self                       0
    link_flair_css_class      15842
    link_flair_text           15841
    locked                        0
    media                     18208
    media_embed                   0
    mobile_ad_url             18208
    name                          0
    num_comments                  0
    original_link             18208
    over_18                       0
    permalink                     0
    post_hint                 14695
    preview                   14695
    promoted                  18208
    promoted_by               18208
    promoted_display_name     18208
    promoted_url              18208
    quarantine                    0
    retrieved_on                  0
    saved                         0
    score                         0
    secure_media              18208
    secure_media_embed            0
    selftext                      0
    spoiler                       0
    stickied                      0
    subreddit                     0
    subreddit_id                  0
    third_party_tracking      18208
    third_party_tracking_2    18208
    thumbnail                     0
    title                         0
    ups                           0
    url                           0
    dtype: int64




```python
# Right away we can see there are many columns with all null values. We
# can drop these right away as they are literally giving us no 
# information.
null_cols = []
for col in df:
    if df[col].isnull().sum() == 18208:
        null_cols.append(col)
    else:
        null_cols = null_cols

df = df.drop(null_cols, axis=1)
```

Now we are left with only columns with either 0 null values, or more than
14,000 null values. To see if these columns with thousands of nulls have
anything of note contained in them, I'm going to go through the value 
counts of each of them.


```python
lots_of_nulls = []
for col in df:
    if df[col].isnull().sum() > 14000:
        lots_of_nulls.append(col)
    else:
        lots_of_nulls = lots_of_nulls
        
for col in lots_of_nulls:
    print(df[col].value_counts()[:10])
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
    re.tar               6
    Trusted              6
    0 0                  6
    Beginner             5
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
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?s=ff05b930089d958e41fd566f183592c6', u'width': 250, u'height': 250}, u'resolutions': [{u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=05f69610841f8c419b3d580bf1b2dac1', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=2abff01ac7255b4a197afb5fb03b0048', u'width': 216, u'height': 216}], u'variants': {}, u'id': u'tEFaKdpbTuSBBWpWQ-kmQ1l_KwNUpQtPpUtOwmLiL-A'}]}                                                                                                                                                                                                                       117
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?s=ff05b930089d958e41fd566f183592c6', u'width': 250, u'height': 250}, u'variants': {}, u'id': u'tEFaKdpbTuSBBWpWQ-kmQ1l_KwNUpQtPpUtOwmLiL-A', u'resolutions': [{u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=05f69610841f8c419b3d580bf1b2dac1', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=2abff01ac7255b4a197afb5fb03b0048', u'width': 216, u'height': 216}]}]}                                                                                                                                                                                                                        52
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?s=5771b03e94f56162bcafaf1079f2a1e7', u'width': 316, u'height': 316}, u'resolutions': [{u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=9189a7b196263db4825e3ab0c948e796', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=c442ba677155996a0a49f51a400124b1', u'width': 216, u'height': 216}], u'variants': {}, u'id': u'nfayPavSUB5ngYv6-19UHNBThsXfcLIDQl4HkEe3Cv0'}]}                                                                                                                                                                                                                        34
    {u'images': [{u'resolutions': [{u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=05f69610841f8c419b3d580bf1b2dac1', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=2abff01ac7255b4a197afb5fb03b0048', u'width': 216, u'height': 216}], u'variants': {}, u'id': u'tEFaKdpbTuSBBWpWQ-kmQ1l_KwNUpQtPpUtOwmLiL-A', u'source': {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?s=ff05b930089d958e41fd566f183592c6', u'width': 250, u'height': 250}}]}                                                                                                                                                                                                                        31
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/dekDuahiNRzoFODVQsNUJKCrIoNBZwElNV5z9YG5rhY.jpg?s=1a0e68c85811600a5df0e8d62e232cb1', u'width': 424, u'height': 210}, u'resolutions': [{u'url': u'https://i.redditmedia.com/dekDuahiNRzoFODVQsNUJKCrIoNBZwElNV5z9YG5rhY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=c190368a97861c5fc52f49e4f26a8a38', u'width': 108, u'height': 53}, {u'url': u'https://i.redditmedia.com/dekDuahiNRzoFODVQsNUJKCrIoNBZwElNV5z9YG5rhY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=0806acb928b986e8c85ec33adbfcfc4d', u'width': 216, u'height': 106}, {u'url': u'https://i.redditmedia.com/dekDuahiNRzoFODVQsNUJKCrIoNBZwElNV5z9YG5rhY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=baab114de58d6cbb36f06ddede6bd7f0', u'width': 320, u'height': 158}], u'variants': {}, u'id': u'waJH_Of06cUPp788N5rg0pLX4aEY6ALwDeEK02fGjMk'}]}     15
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?s=ff05b930089d958e41fd566f183592c6', u'width': 250, u'height': 250}, u'variants': {}, u'resolutions': [{u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=05f69610841f8c419b3d580bf1b2dac1', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/apn0jimBDcu7v5NrQv4_AjqWqHWQMnATxOAbLbDgyQw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=2abff01ac7255b4a197afb5fb03b0048', u'width': 216, u'height': 216}], u'id': u'tEFaKdpbTuSBBWpWQ-kmQ1l_KwNUpQtPpUtOwmLiL-A'}]}                                                                                                                                                                                                                        13
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?s=5771b03e94f56162bcafaf1079f2a1e7', u'width': 316, u'height': 316}, u'variants': {}, u'id': u'nfayPavSUB5ngYv6-19UHNBThsXfcLIDQl4HkEe3Cv0', u'resolutions': [{u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=9189a7b196263db4825e3ab0c948e796', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/yzSfTlKTSYGpEXeFgyDvHlfoLGOFQJqPuH_Y38RBz2U.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=c442ba677155996a0a49f51a400124b1', u'width': 216, u'height': 216}]}]}                                                                                                                                                                                                                        11
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/qMvjJDOF4LGT3ZPgIk9Rojl9rjH9R9zAvOu6H27TRE4.jpg?s=56de55667185c86831aaec972eddb90b', u'width': 50, u'height': 50}, u'resolutions': [], u'variants': {}, u'id': u'BN6GABrPfvOiNpUg3pX8VtN0SSwRwgC82edusQp5JPw'}]}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 8
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/23VgbE0_fYbx6bXst2WVRO-5A1J43T9GliZp4XJgB3I.jpg?s=2c0cf9c40a19c3b7836fa1092e8f41e5', u'width': 75, u'height': 75}, u'variants': {}, u'id': u'CYq1eVd12A7_YNDyWTMK6OAXlBSv5qQmV6vFGYOE5rE', u'resolutions': []}]}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 6
    {u'images': [{u'source': {u'url': u'https://i.redditmedia.com/8pkEP0b3dVtOhHA0mfsVcYgtCTwtxX-F-ocIZmubs9o.jpg?s=6eaf888f600951dd66ee368596568207', u'width': 250, u'height': 250}, u'resolutions': [{u'url': u'https://i.redditmedia.com/8pkEP0b3dVtOhHA0mfsVcYgtCTwtxX-F-ocIZmubs9o.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=54a75fa1a48fa1c13cf6b85c0d6c9d19', u'width': 108, u'height': 108}, {u'url': u'https://i.redditmedia.com/8pkEP0b3dVtOhHA0mfsVcYgtCTwtxX-F-ocIZmubs9o.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=62a696e58983d5eaa3293a15ade87a7b', u'width': 216, u'height': 216}], u'variants': {}, u'id': u'y40yRHlOMU6vJxbOuQe-jhLF_xtFKZ6AHmC7r_hqi74'}]}                                                                                                                                                                                                                         6
    Name: preview, dtype: int64


The only thing of note I can see in these columns is that the 'distinguished' column is only used to designate if a moderator of that subreddit is making the post. There's also the 'link_flair_css_class' and 'link_flair_text' columns which are mostly used to categorize the type of post, but as they both have so many null values I'm assuming that only specific subreddits use the flairs. Because of this, the only use from all of these columns with more than 20,000 null values will be to make a column designating whether a post was made by a moderator or not.


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
# Taking a look into the values in the boolean columns to see if there is anything useful
# to be gleaned from them.
for col in boolean_columns:
    print(df[col].value_counts())
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
df['nsfw'] = df['thumbnail'].apply(lambda x: 1 if x == 'nsfw' else 0)
```


```python
# Drop all columns with irrelevant or duplicate information as outlined
# above.
df = df.drop(['author', 'created_utc', 'domain', 'downs', 'edited', 'gilded', 'id', 
              'media_embed', 'name', 'permalink', 'retrieved_on', 'secure_media_embed', 
              'subreddit_id', 'thumbnail', 'ups', 'url'], axis=1)
```


```python
# Create a column called 'selftext_length' with the length of the text of the post. Make sure
# that entries that are deleted or removed have values of 0 instead of 9.
df['selftext_length'] = df['selftext'].apply(lambda x: len(x))
```


```python
corr = df[['num_comments', 'score', 'selftext_length']].corr()

fig, ax = plt.subplots(figsize=(6,6))

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

ax = sns.heatmap(corr, mask=mask, ax=ax, annot=True)

ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=10)
ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=10)

plt.show()
```


![png](/images/Submission_files/Submission_30_0.png)



```python
# As we can see, the number of comments and the score are very highly correlated. While the
# length of the selftext is not at all correlated with either of them. I will keep both the
# score and the number of comments in the model for now, as I'm assuming that because most of
# my processing will be done with NLP I will end up with thousands of predictor columns anyways
# and have to use a model that is very robust to overfitting. I am happy with any additional
# variance a datapoint may be able to add.
```


```python
df.head()
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
      <th>selftext</th>
      <th>subreddit</th>
      <th>title</th>
      <th>moderator</th>
      <th>nsfw</th>
      <th>selftext_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>I have a Sony surround sound system for a blu-...</td>
      <td>techsupport</td>
      <td>Help with audio set-up</td>
      <td>0</td>
      <td>0</td>
      <td>212</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.0</td>
      <td>23.0</td>
      <td>I've written what seems to be a prohibitively ...</td>
      <td>learnprogramming</td>
      <td>Optimizing code for speed</td>
      <td>0</td>
      <td>0</td>
      <td>5546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5.0</td>
      <td>12.0</td>
      <td>I'm writing an article called "Video Games Tha...</td>
      <td>gamedev</td>
      <td>Seeking Tales of Development Woe (and Triumph)...</td>
      <td>0</td>
      <td>0</td>
      <td>1240</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.0</td>
      <td>6.0</td>
      <td>I have the following representation of argumen...</td>
      <td>learnpython</td>
      <td>currying functions using functools</td>
      <td>0</td>
      <td>0</td>
      <td>917</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>I am about to create a website where users use...</td>
      <td>learnprogramming</td>
      <td>Text Editor integration</td>
      <td>0</td>
      <td>0</td>
      <td>628</td>
    </tr>
  </tbody>
</table>
</div>



Our dataset is now completely clean and contains only the columns that I believe will be relevant to predicting the subreddit of the post. We can now move onto the Natural Language Processing step in order to change the 'selftext' and 'title' columns into predictors that are actually usable for modeling.

## Natural Language Processing

Before we start onto changing our text data into data that is useful for modeling, we will have to split up our data into a training set and a test set. The values we are trying to predict are the subreddits, which we will need to change into numerical values, and our predictors will be everything else.


```python
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
```


```python
# Creates a dictionary of all the subreddits in the dataset with an integer attached that will
# correspond to only that subreddit.

subreddit_dictionary = {}
for ind, val in enumerate (df['subreddit'].value_counts().index.tolist()):
    subreddit_dictionary[val] = ind
```


```python
# Making our target column of integers correlating to subreddits by referencing the dictionary
# created above.

y = df['subreddit'].apply(lambda x: subreddit_dictionary[x])
```


```python
# Making our predixtor columns to be the cleaned dataset minus just the target column we have
# already extracted to y.

df_notarget = df.drop('subreddit', axis=1)
X = df_notarget
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2017)
```

We now have our separate training and testing datasets. From here on out, everything we do to the training set to get it ready for modeling will also need to be applied to the testing set. We can do this in one fell swoop at the end of our preprocessing stage, before we get into the modeling.


```python
# Before we start into the Natural Language Processing we will need to extract the text columns
# ('selftext' and 'title') from both X_train and X_test into their own separate series. This is
# so that we have separate values for words contained within the title of a post and within the
# text of a post, which should be an important distinction.

X_train_selftext = X_train['selftext']
X_train_title = X_train['title']
X_test_selftext = X_test['selftext']
X_test_title = X_test['title']
```


```python
X_train
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
      <th>selftext</th>
      <th>title</th>
      <th>moderator</th>
      <th>nsfw</th>
      <th>selftext_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20917</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>My dad has a workstation PC as well as an inte...</td>
      <td>Boot sequence errors on old computer with mast...</td>
      <td>0</td>
      <td>0</td>
      <td>2479</td>
    </tr>
    <tr>
      <th>21670</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>Hello. I just reinstalled my windows 10 instal...</td>
      <td>7 hours do defrag.</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
    </tr>
    <tr>
      <th>16748</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>I recently order a gtx 1060 (gigabyte g1 gamin...</td>
      <td>Is this an adequate power supply for my build?</td>
      <td>0</td>
      <td>0</td>
      <td>525</td>
    </tr>
    <tr>
      <th>5899</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>Hi! So I'm gonna be honest and make it maybe r...</td>
      <td>How to invert the fn keys</td>
      <td>0</td>
      <td>0</td>
      <td>674</td>
    </tr>
    <tr>
      <th>22327</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>Should I place them all in a Protocols.swift f...</td>
      <td>Swift style guidelines: where to place protoco...</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
    </tr>
    <tr>
      <th>983</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>The problem is that I don't have a working dri...</td>
      <td>Need urgent help. Got a new motherboard, but n...</td>
      <td>0</td>
      <td>0</td>
      <td>252</td>
    </tr>
    <tr>
      <th>4455</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>turn on computer to find my bios are trying to...</td>
      <td>/BIOS is updating/ stalled</td>
      <td>0</td>
      <td>0</td>
      <td>590</td>
    </tr>
    <tr>
      <th>16743</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>Not homework just a personal project  \nso I'v...</td>
      <td>md5 bruteforcing, need help with adding salt.</td>
      <td>0</td>
      <td>0</td>
      <td>416</td>
    </tr>
    <tr>
      <th>23765</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>I built a website that uses bootstrap-tables. ...</td>
      <td>Slow on Android, fast on everything else?</td>
      <td>0</td>
      <td>0</td>
      <td>280</td>
    </tr>
    <tr>
      <th>25579</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>I've had a Samsung 840 Evo 250GB plugged into ...</td>
      <td>SSD constantly disconnecting/reconnecting</td>
      <td>0</td>
      <td>0</td>
      <td>1048</td>
    </tr>
    <tr>
      <th>25739</th>
      <td>19.0</td>
      <td>16.0</td>
      <td>Building a scraper that would run 5 days a wee...</td>
      <td>Made something have no idea how to deploy it t...</td>
      <td>0</td>
      <td>0</td>
      <td>496</td>
    </tr>
    <tr>
      <th>20242</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>I am making an app where a user can add a bunc...</td>
      <td>Firebase storing/retrieving multiple copies of...</td>
      <td>0</td>
      <td>0</td>
      <td>951</td>
    </tr>
    <tr>
      <th>19933</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>Sorry if the title is ambiguous. I wasn't sure...</td>
      <td>[Python] Trying to create a program that backu...</td>
      <td>0</td>
      <td>0</td>
      <td>928</td>
    </tr>
    <tr>
      <th>7477</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>The course that I TA for has a new professor s...</td>
      <td>TA'ing a course that's changing languages...be...</td>
      <td>0</td>
      <td>0</td>
      <td>297</td>
    </tr>
    <tr>
      <th>1362</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>I want to learn to code a game using visual st...</td>
      <td>I want to learn to code a game using visual st...</td>
      <td>0</td>
      <td>0</td>
      <td>683</td>
    </tr>
    <tr>
      <th>936</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>To all the developers here: I would love to he...</td>
      <td>Ingredients for the perfect game studio?</td>
      <td>0</td>
      <td>0</td>
      <td>661</td>
    </tr>
    <tr>
      <th>13818</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>http://imgur.com/a/Ysv1K\n\nI dont know why th...</td>
      <td>This pops up whenever i start MS word, how can...</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
    </tr>
    <tr>
      <th>13589</th>
      <td>1.0</td>
      <td>0.0</td>
      <td>Ok folks hear me out on this, I have bulk stan...</td>
      <td>Dumb Computer Game Question</td>
      <td>0</td>
      <td>0</td>
      <td>397</td>
    </tr>
    <tr>
      <th>4228</th>
      <td>14.0</td>
      <td>1.0</td>
      <td>I used my Turtle Beach PX3 headset as a record...</td>
      <td>Computer no longer shows microphone</td>
      <td>0</td>
      <td>0</td>
      <td>623</td>
    </tr>
    <tr>
      <th>890</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>Firstly I appreciate anyone who takes the time...</td>
      <td>ArrayDictionary Class</td>
      <td>0</td>
      <td>0</td>
      <td>15840</td>
    </tr>
    <tr>
      <th>2994</th>
      <td>19.0</td>
      <td>2.0</td>
      <td>I recently bought a GTX 1070 (MSI) and I've be...</td>
      <td>I have a 1070 and I'm getting shitty frames. P...</td>
      <td>0</td>
      <td>0</td>
      <td>1025</td>
    </tr>
    <tr>
      <th>25531</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>I am creating a flask web app that is based on...</td>
      <td>When to look into task queues?</td>
      <td>0</td>
      <td>0</td>
      <td>881</td>
    </tr>
    <tr>
      <th>13771</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>In case you want to help out with this thread,...</td>
      <td>Analyzing Crash Bandicoot 2 mechanics with Jav...</td>
      <td>0</td>
      <td>0</td>
      <td>2694</td>
    </tr>
    <tr>
      <th>14932</th>
      <td>16.0</td>
      <td>0.0</td>
      <td>So to make it short I wanted to change motherb...</td>
      <td>Huge problem</td>
      <td>0</td>
      <td>0</td>
      <td>680</td>
    </tr>
    <tr>
      <th>4693</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>Few pages in my app have certain animations in...</td>
      <td>No back button on certain pages</td>
      <td>0</td>
      <td>0</td>
      <td>279</td>
    </tr>
    <tr>
      <th>8587</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>This Dell laptop beeps in a series of 3 times ...</td>
      <td>Dell Inspiron 11 3000 Series 2-in-1 laptop bee...</td>
      <td>0</td>
      <td>0</td>
      <td>1602</td>
    </tr>
    <tr>
      <th>4610</th>
      <td>2.0</td>
      <td>20.0</td>
      <td>https://www.youtube.com/watch?v=LvUHrCj-beY  \...</td>
      <td>If you're struggling with basic Python concept...</td>
      <td>0</td>
      <td>0</td>
      <td>744</td>
    </tr>
    <tr>
      <th>15406</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>I have a genuine copy of windows just to point...</td>
      <td>My wallpaper keeps turning into a black screen...</td>
      <td>0</td>
      <td>0</td>
      <td>545</td>
    </tr>
    <tr>
      <th>14635</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>I've put a background image and changed the op...</td>
      <td>boxes behind all incoming posts so everyone ca...</td>
      <td>0</td>
      <td>0</td>
      <td>150</td>
    </tr>
    <tr>
      <th>3734</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>I built a new computer and have been having so...</td>
      <td>Is this an issue with my graphics card or my P...</td>
      <td>0</td>
      <td>0</td>
      <td>891</td>
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
    </tr>
    <tr>
      <th>25722</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>I will be only be able to buy one 1 TB hard di...</td>
      <td>Best way to partitions 1TB hard disk for gaming?</td>
      <td>0</td>
      <td>0</td>
      <td>207</td>
    </tr>
    <tr>
      <th>20187</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>Last year i bought a windows gaming computer. ...</td>
      <td>AMD FreeSync useless?</td>
      <td>0</td>
      <td>0</td>
      <td>306</td>
    </tr>
    <tr>
      <th>19307</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>Hey guys,\n\nI'm currently planning to "reset"...</td>
      <td>Mainboard drivers - where to get them in one p...</td>
      <td>0</td>
      <td>0</td>
      <td>1039</td>
    </tr>
    <tr>
      <th>5287</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>I have the windows 10 iso file and other hdd p...</td>
      <td>install windows to other HDD</td>
      <td>0</td>
      <td>0</td>
      <td>87</td>
    </tr>
    <tr>
      <th>26254</th>
      <td>3.0</td>
      <td>6.0</td>
      <td>I have a large volume of data, 16TB personal, ...</td>
      <td>Long Term Data Storage for Consumer</td>
      <td>0</td>
      <td>0</td>
      <td>1121</td>
    </tr>
    <tr>
      <th>5266</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>Currently I am using my WiFi to write this pos...</td>
      <td>WiFi not working for most tasks, however it wo...</td>
      <td>0</td>
      <td>0</td>
      <td>768</td>
    </tr>
    <tr>
      <th>19170</th>
      <td>16.0</td>
      <td>1.0</td>
      <td>So, this doesn't seem to be an isolated issue....</td>
      <td>SSD filling up- usual options exhausted</td>
      <td>0</td>
      <td>0</td>
      <td>1102</td>
    </tr>
    <tr>
      <th>9687</th>
      <td>5.0</td>
      <td>0.0</td>
      <td>Hey, I'm a beginner in programming, trying to ...</td>
      <td>Java and Classes</td>
      <td>0</td>
      <td>0</td>
      <td>437</td>
    </tr>
    <tr>
      <th>18397</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>Hola,\n\nIt has been a few while since I used ...</td>
      <td>Windows 10 Bluetooth No Longer Detecting Devices</td>
      <td>0</td>
      <td>0</td>
      <td>423</td>
    </tr>
    <tr>
      <th>13798</th>
      <td>40.0</td>
      <td>11.0</td>
      <td>Could I build a productivity app for my own pe...</td>
      <td>Can I build an app for my iPhone specifically ...</td>
      <td>0</td>
      <td>0</td>
      <td>210</td>
    </tr>
    <tr>
      <th>3559</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>I'm going to start coding c++ and c# again, be...</td>
      <td>Visual studio vs codeblocks vs kdevelop?</td>
      <td>0</td>
      <td>0</td>
      <td>759</td>
    </tr>
    <tr>
      <th>24389</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>The sound doesn't seem to function properly. W...</td>
      <td>HyperX Cloud I HeadPhone's sounds aren't funct...</td>
      <td>0</td>
      <td>0</td>
      <td>205</td>
    </tr>
    <tr>
      <th>4398</th>
      <td>28.0</td>
      <td>0.0</td>
      <td>- Sketch\n- Screenhero (will eventually be int...</td>
      <td>4 apps holding me back from going 100% Linux</td>
      <td>0</td>
      <td>0</td>
      <td>184</td>
    </tr>
    <tr>
      <th>19702</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>Hello,\nI am looking for a backup software tha...</td>
      <td>Automatic File/ Folder Backup Programs</td>
      <td>0</td>
      <td>0</td>
      <td>349</td>
    </tr>
    <tr>
      <th>12248</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>I have about 250 folders that are duplicated w...</td>
      <td>Moving multiple folders within folders</td>
      <td>0</td>
      <td>0</td>
      <td>557</td>
    </tr>
    <tr>
      <th>11666</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>I have a JVC DVD Digital Theater System TH-G40...</td>
      <td>JVC Stereo System Help</td>
      <td>0</td>
      <td>0</td>
      <td>225</td>
    </tr>
    <tr>
      <th>1419</th>
      <td>5.0</td>
      <td>5.0</td>
      <td>I want to develop a calculator with the help o...</td>
      <td>What are some good resources to start with PyQt?</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
    </tr>
    <tr>
      <th>18869</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>I would like to print a christmas card, but do...</td>
      <td>Card stock for my Epson workforce 2650</td>
      <td>0</td>
      <td>0</td>
      <td>259</td>
    </tr>
    <tr>
      <th>16581</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>Hello,\n\nMy computer's screen 'splits' the sc...</td>
      <td>Screen split straight down the middle after pl...</td>
      <td>0</td>
      <td>0</td>
      <td>384</td>
    </tr>
    <tr>
      <th>14101</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>I'm new to anything that isn't Ubuntu, so I'm ...</td>
      <td>Installed Solus Shannon, Can't VNC or SSH into...</td>
      <td>0</td>
      <td>0</td>
      <td>377</td>
    </tr>
    <tr>
      <th>25573</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>Ok so my computer (windows 10 64 bit) had been...</td>
      <td>Partmgr.sys is corrupt</td>
      <td>0</td>
      <td>0</td>
      <td>801</td>
    </tr>
    <tr>
      <th>1743</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>I love my laptop but it would be perfect if I ...</td>
      <td>Lenovo g50-70 screen and keyboard upgrade?</td>
      <td>0</td>
      <td>0</td>
      <td>823</td>
    </tr>
    <tr>
      <th>19654</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>im making a pokemon game but im having an issu...</td>
      <td>help with print and instances[python]</td>
      <td>0</td>
      <td>0</td>
      <td>210</td>
    </tr>
    <tr>
      <th>11449</th>
      <td>2.0</td>
      <td>0.0</td>
      <td>title</td>
      <td>Sophomore in CS and striving to be a Network E...</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>13332</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>Just to share: been discovering the power of n...</td>
      <td>Discovering numpy masked arrays</td>
      <td>0</td>
      <td>0</td>
      <td>655</td>
    </tr>
    <tr>
      <th>23784</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>I was redirected here from /r/python, I hope y...</td>
      <td>How hard would it be for a python beginner to ...</td>
      <td>0</td>
      <td>0</td>
      <td>1227</td>
    </tr>
    <tr>
      <th>7030</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>So a while ago, my PC started freezing all of ...</td>
      <td>PC freezing while playing games?</td>
      <td>0</td>
      <td>0</td>
      <td>826</td>
    </tr>
    <tr>
      <th>20338</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>Hello guys,\n\nI have been coding for about 10...</td>
      <td>Developing algorithm understanding and pursuin...</td>
      <td>0</td>
      <td>0</td>
      <td>1307</td>
    </tr>
    <tr>
      <th>15194</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>Hey everyone, I'm pretty sure I know what the ...</td>
      <td>Swapped to new case, but get no display.</td>
      <td>0</td>
      <td>0</td>
      <td>1140</td>
    </tr>
    <tr>
      <th>14522</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>Hello whoever reads this!\n\nSo sorry to be a ...</td>
      <td>Laptop Upgrading Blues</td>
      <td>0</td>
      <td>0</td>
      <td>861</td>
    </tr>
  </tbody>
</table>
<p>14566 rows × 7 columns</p>
</div>




```python
print(X_train_selftext)

# From looking through a bit of these 'selftext' values, we can see that line breaks in the
# text are signified with '\n'. We will replace these with a space because sometimes there is a
# linebreak with no spaces inbetween the two lines, meaning if we replaced it with an empty 
# string it may combine two words together. We will do this to both the testing and training
# sets.

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
    25722    I will be only be able to buy one 1 TB hard di...
    20187    Last year i bought a windows gaming computer. ...
    19307    Hey guys,\n\nI'm currently planning to "reset"...
    5287     I have the windows 10 iso file and other hdd p...
    26254    I have a large volume of data, 16TB personal, ...
    5266     Currently I am using my WiFi to write this pos...
    19170    So, this doesn't seem to be an isolated issue....
    9687     Hey, I'm a beginner in programming, trying to ...
    18397    Hola,\n\nIt has been a few while since I used ...
    13798    Could I build a productivity app for my own pe...
    3559     I'm going to start coding c++ and c# again, be...
    24389    The sound doesn't seem to function properly. W...
    4398     - Sketch\n- Screenhero (will eventually be int...
    19702    Hello,\nI am looking for a backup software tha...
    12248    I have about 250 folders that are duplicated w...
    11666    I have a JVC DVD Digital Theater System TH-G40...
    1419     I want to develop a calculator with the help o...
    18869    I would like to print a christmas card, but do...
    16581    Hello,\n\nMy computer's screen 'splits' the sc...
    14101    I'm new to anything that isn't Ubuntu, so I'm ...
    25573    Ok so my computer (windows 10 64 bit) had been...
    1743     I love my laptop but it would be perfect if I ...
    19654    im making a pokemon game but im having an issu...
    11449                                                title
    13332    Just to share: been discovering the power of n...
    23784    I was redirected here from /r/python, I hope y...
    7030     So a while ago, my PC started freezing all of ...
    20338    Hello guys,\n\nI have been coding for about 10...
    15194    Hey everyone, I'm pretty sure I know what the ...
    14522    Hello whoever reads this!\n\nSo sorry to be a ...
    Name: selftext, Length: 14566, dtype: object



```python
# We will be creating a text preprocesser in order to best turn our text data into something
# that is usable for modeling. We will be using the PorterStemmer to combine all derivations of
# root words into just the root, and stopwords to remove meaningless words from the text.

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string
```


```python
# Creating our text preprocesser. The cleaner function will remove punctuation and numbers from
# the text, change it all into lowercase, and add all words that are not in the english stop-
# words list to a list of final words, while stemming words with the same roots together.

def text_preprocesser(text):
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


```python
# Because our text_preprocesser we just created above strips all punctuation and digits 
# from the text we run it through, I will manually create binary classification columns 
# for where the title and text of a post contains punctuation or digits of some sort.

X_train['title_contains_?'] = X_train_title.str.contains('\?').astype(int)
X_train['title_contains_['] = X_train_title.str.contains('\[').astype(int)
X_train['title_contains_C++'] = X_train_title.str.contains('[Cc]\+\+').astype(int)
X_train['title_contains_Windows7or8'] = X_train_title.str.contains('[Ww]indows [78]').astype(int)
X_train['title_contains_Windows10'] = X_train_title.str.contains('[Ww]indows 10').astype(int)
X_train['title_contains_64bit'] = X_train_title.str.contains('64[\s\S][Bb]it').astype(int)
X_train['title_contains_32bit'] = X_train_title.str.contains('32[\s\S][Bb]it').astype(int)
X_train['title_contains_64bit'] = X_train_title.str.contains('64[\s\S][Bb]it').astype(int)
X_train['title_contains_()'] = X_train_title.str.contains('\(\)').astype(int)
X_train['title_contains_underscore'] = X_train_title.str.contains('\_').astype(int)
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # Remove the CWD from sys.path while we load stuff.
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # This is added back by InteractiveShellApp.init_path()
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if sys.path[0] == '':
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:13: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      del sys.path[0]
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      



```python
X_train['selftext_contains_?'] = X_train_selftext.str.contains('\?').astype(int)
X_train['selftext_contains_['] = X_train_selftext.str.contains('\[').astype(int)
X_train['selftext_contains_C++'] = X_train_selftext.str.contains('[Cc]\+\+').astype(int)
X_train['selftext_contains_Windows7or8'] = X_train_selftext.str.contains('[Ww]indows [78]').astype(int)
X_train['selftext_contains_Windows10'] = X_train_selftext.str.contains('[Ww]indows 10').astype(int)
X_train['selftext_contains_64bit'] = X_train_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_train['selftext_contains_32bit'] = X_train_selftext.str.contains('32[\s\S][Bb]it').astype(int)
X_train['selftext_contains_64bit'] = X_train_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_train['selftext_contains_()'] = X_train_selftext.str.contains('\(\)').astype(int)
X_train['selftext_contains_underscore'] = X_train_selftext.str.contains('\_').astype(int)
```


```python
# Making the same columns as above for the test data

X_test['title_contains_?'] = X_test_title.str.contains('\?').astype(int)
X_test['title_contains_['] = X_test_title.str.contains('\[').astype(int)
X_test['title_contains_C++'] = X_test_title.str.contains('[Cc]\+\+').astype(int)
X_test['title_contains_Windows7or8'] = X_test_title.str.contains('[Ww]indows [78]').astype(int)
X_test['title_contains_Windows10'] = X_test_title.str.contains('[Ww]indows 10').astype(int)
X_test['title_contains_64bit'] = X_test_title.str.contains('64[\s\S][Bb]it').astype(int)
X_test['title_contains_32bit'] = X_test_title.str.contains('32[\s\S][Bb]it').astype(int)
X_test['title_contains_64bit'] = X_test_title.str.contains('64[\s\S][Bb]it').astype(int)
X_test['title_contains_()'] = X_test_title.str.contains('\(\)').astype(int)
X_test['title_contains_underscore'] = X_test_title.str.contains('_').astype(int)
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      after removing the cwd from sys.path.
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:5: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      """
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if __name__ == '__main__':
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # Remove the CWD from sys.path while we load stuff.
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      # This is added back by InteractiveShellApp.init_path()
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/ipykernel_launcher.py:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      if sys.path[0] == '':



```python
X_test['selftext_contains_?'] = X_test_selftext.str.contains('\?').astype(int)
X_test['selftext_contains_['] = X_test_selftext.str.contains('\[').astype(int)
X_test['selftext_contains_C++'] = X_test_selftext.str.contains('[Cc]\+\+').astype(int)
X_test['selftext_contains_Windows7or8'] = X_test_selftext.str.contains('[Ww]indows [78]').astype(int)
X_test['selftext_contains_Windows10'] = X_test_selftext.str.contains('[Ww]indows 10').astype(int)
X_test['selftext_contains_64bit'] = X_test_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_test['selftext_contains_32bit'] = X_test_selftext.str.contains('32[\s\S][Bb]it').astype(int)
X_test['selftext_contains_64bit'] = X_test_selftext.str.contains('64[\s\S][Bb]it').astype(int)
X_test['selftext_contains_()'] = X_test_selftext.str.contains('\(\)').astype(int)
X_test['selftext_contains_underscore'] = X_test_selftext.str.contains('\_').astype(int)
```

### Count Vectorizer


```python
# Use CountVectorizer and our text-preprocesser from above to process our selftext data. We
# will also limit the words chosen to only those that occur in a minimum of 1/1000th of the 
# entries, and a maximum of half of the entries.

cv = CountVectorizer(preprocessor=text_preprocesser, min_df=0.001, max_df=0.50)
train_selftext_cv = cv.fit(X_train_selftext)

# Because it's almost a certainty that there will be shared common words between the selftext
# and the title of a post, we will need to create feature names with the prefix 'selftext_' to
# make sure that we get separate entries.

selftext_feature_names = []
for feature in cv.get_feature_names():
    selftext_feature_names.append('selftext_' + feature)

# Create a dataframe from the CountVectorizer processed data using the feature names from above

train_selftext_cv_df = pd.DataFrame(cv.transform(X_train_selftext).todense(), 
                                          columns=selftext_feature_names)

train_selftext_cv_df.shape
```




    (14566, 3323)




```python
# Now do the same as above for our testing selftext data.

test_selftext_cv_df = pd.DataFrame(cv.transform(X_test_selftext).todense(), 
                                  columns=selftext_feature_names)
test_selftext_cv_df.shape
```




    (3642, 3323)




```python
# Repeat the same processes from the above two steps for our title data, making sure to add
# the prefix of 'title_' to our features produced here. We are printing out the shapes of our
# two new dataframes just to confirm that everything has went as planned.

cv = CountVectorizer(preprocessor=text_preprocesser, min_df=0.001, max_df=0.50)
train_title_cv = cv.fit(X_train_title)

title_feature_names = []
for feature in cv.get_feature_names():
    title_feature_names.append('title_' + feature)

train_title_cv_df = pd.DataFrame(cv.transform(X_train_title).todense(), 
                                          columns=title_feature_names)

test_title_cv_df = pd.DataFrame(cv.transform(X_test_title).todense(),
                               columns=title_feature_names)

print(train_title_cv_df.shape, '\n', test_title_cv_df.shape)
```

    (14566, 981) 
     (3642, 981)


#### TFIDF Vectorizer

Using pretty much the same process as we used for Count Vectorizer, we will also process our text with Tfidf Vectorizer and cut down our number of features using Truncated SVD, a method of PCA best used on the sparse matrices output through NLP.


```python
from sklearn.decomposition import TruncatedSVD
```

#### Selftext


```python
tfidf = TfidfVectorizer(preprocessor=text_preprocesser)
train_selftext_tfidf = tfidf.fit(X_train_selftext)

selftext_feature_names = []
for feature in tfidf.get_feature_names():
    selftext_feature_names.append('selftext_' + feature)

train_selftext_tfidf_df = pd.DataFrame(tfidf.transform(X_train_selftext).todense(), 
                                          columns=selftext_feature_names)

test_selftext_tfidf_df = pd.DataFrame(tfidf.transform(X_test_selftext).todense(),
                               columns=selftext_feature_names)

print(train_selftext_tfidf_df.shape, '\n', test_selftext_tfidf_df.shape)
```

    (14566, 49629) 
     (3642, 49629)


Tfidf Vectorizer created 49629 columns from our selftext text, we'll try cutting that down to 2500 using Truncated SVD, and see how much of the variance in the data is explained with only these heavily reduced amount of columns.


```python
selftext_tsvd = TruncatedSVD(n_components=2500)
selftext_tsvd.fit(train_selftext_tfidf_df.values)
```




    TruncatedSVD(algorithm='randomized', n_components=2500, n_iter=5,
           random_state=None, tol=0.0)




```python
plt.plot(range(2500), selftext_tsvd.explained_variance_ratio_.cumsum())

plt.title('Selftext TSVD', fontsize=20)
plt.xlabel('Number of Components in Truncated SVD', fontsize=14)
plt.ylabel('Percentage of Expained Variance', fontsize=14)

# About 80% of the variance in the selftext data is explained with only 2500 of the original
# 49629 columns, this should save A LOT of time while modeling and not have too much of an
# impact on our model
```




    <matplotlib.text.Text at 0x10caf54a8>




![png](/images/Submission_files/Submission_62_1.png)



```python
X_train_selftext_tfidf = selftext_tsvd.transform(train_selftext_tfidf_df.values)
X_test_selftext_tfidf = selftext_tsvd.transform(test_selftext_tfidf_df.values)
```


```python
# Creating dataframes from our transformed selftext data, making sure to give them a list
# of column names. I will give the transformed title data column names continuing on from 2500
# so that when I go to merge the dataframes all back together there is no overlap. I am also
# setting the index to the index of X_train as the NLP transformations have reset them back to
# ascending from 0 as opposed to the specific rows that train_test_split chose for X_train.

X_train_selftext_tfidf_df = pd.DataFrame(X_train_selftext_tfidf, 
                                        columns=list(range(0, 2500)),
                                        index=X_train.index)
X_test_selftext_tfidf_df = pd.DataFrame(X_test_selftext_tfidf,
                                       columns=list(range(0, 2500)),
                                       index=X_test.index)
```

#### Title


```python
tfidf = TfidfVectorizer(preprocessor=text_preprocesser)
train_title_tfidf = tfidf.fit(X_train_title)

title_feature_names = []
for feature in tfidf.get_feature_names():
    title_feature_names.append('title_' + feature)

train_title_tfidf_df = pd.DataFrame(tfidf.transform(X_train_title).todense(), 
                                          columns=title_feature_names)

test_title_tfidf_df = pd.DataFrame(tfidf.transform(X_test_title).todense(),
                               columns=title_feature_names)

print(train_title_tfidf_df.shape, '\n', test_title_tfidf_df.shape)
```

    (14566, 8723) 
     (3642, 8723)


Tfidf Vectorizer created 8723 columns from our title text, we'll try cutting that down to 1000 using Truncated SVD.


```python
title_tsvd = TruncatedSVD(n_components=1000)
title_tsvd.fit(train_title_tfidf_df.values)
```




    TruncatedSVD(algorithm='randomized', n_components=1000, n_iter=5,
           random_state=None, tol=0.0)




```python
plt.plot(range(1000), title_tsvd.explained_variance_ratio_.cumsum())

plt.title('Title TSVD', fontsize=20)
plt.xlabel('Number of Components in Truncated SVD', fontsize=14)
plt.ylabel('Percentage of Expained Variance', fontsize=14)

# About 70% of the variance in the title data is explained with only 1000 of the original
# 8723 columns.
```




    <matplotlib.text.Text at 0x1a855c02e8>




![png](/images/Submission_files/Submission_69_1.png)



```python
X_train_title_tfidf = title_tsvd.transform(train_title_tfidf_df.values)
X_test_title_tfidf = title_tsvd.transform(test_title_tfidf_df.values)
```


```python
X_train_title_tfidf_df = pd.DataFrame(X_train_title_tfidf,
                                     columns=list(range(2500,3500)),
                                     index=X_train.index)
X_test_title_tfidf_df = pd.DataFrame(X_test_title_tfidf,
                                    columns=list(range(2500,3500)),
                                    index=X_test.index)
```

## Getting Ready for Modeling

Now that we have processed all our our text data and have it in 4 separate dataframes (selftext and title for both the training and the testing set) we will need to put all of our processed data back together into just one dataframe each for both X_train and X_test. However, since we have transformed our data with NLP using both CountVectorizer and TfidfVectorizer, I will create an X_train and X_test dataframe for each of the two methods.

However, before we can do that we will want to scale our numerical data that we have not done anything with yet.


```python
# We can safely drop our text columns from both X_train and X_test as they have already been
# extracted and processed.

X_train = X_train.drop(['selftext', 'title'], axis=1)
X_test = X_test.drop(['selftext', 'title'], axis=1)
```


```python
X_train.head()

# We can see that we have 23 numerical columns that we want to use in our model. Because 20 
# of them are binary classification columns, we will not need to scale those, but the other 
# three will definitely benefit from scaling.
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
      <th>moderator</th>
      <th>nsfw</th>
      <th>selftext_length</th>
      <th>title_contains_?</th>
      <th>title_contains_[</th>
      <th>title_contains_C++</th>
      <th>title_contains_Windows7or8</th>
      <th>title_contains_Windows10</th>
      <th>title_contains_64bit</th>
      <th>title_contains_32bit</th>
      <th>title_contains_()</th>
      <th>title_contains_underscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20917</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>2479</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21670</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>214</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16748</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>525</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5899</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>674</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22327</th>
      <td>2.0</td>
      <td>2.0</td>
      <td>0</td>
      <td>0</td>
      <td>117</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_train[['num_comments', 'score', 'selftext_length']].describe()

# From looking at the description of our 3 non-binary-classification columns, we can see that
# there some extreme outliers in all 3. We will want to use a scaler that is very robust to
# outliers, and for that reason we will be using QuantileTransformer to completely negate the
# outliers in the data.
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




```python
from sklearn.preprocessing import RobustScaler, QuantileTransformer
```


```python
# Here we are scaling our three columns using QuantileTransformer and replacing the original
# values in X_train with the scaled values.

qt = QuantileTransformer()
X_train_qt = qt.fit_transform(X_train[['num_comments', 'score', 'selftext_length']])

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
# Performing the same scaling as above to the relevant testing data columns.

X_test_qt = qt.transform(X_test[['num_comments', 'score', 'selftext_length']])

X_test[['num_comments', 'score', 'selftext_length']] = X_test_qt
```


```python
X_train.head()
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
      <th>moderator</th>
      <th>nsfw</th>
      <th>selftext_length</th>
      <th>title_contains_?</th>
      <th>title_contains_[</th>
      <th>title_contains_C++</th>
      <th>title_contains_Windows7or8</th>
      <th>title_contains_Windows10</th>
      <th>title_contains_64bit</th>
      <th>title_contains_32bit</th>
      <th>title_contains_()</th>
      <th>title_contains_underscore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20917</th>
      <td>0.226727</td>
      <td>0.371872</td>
      <td>0</td>
      <td>0</td>
      <td>0.971441</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21670</th>
      <td>0.835335</td>
      <td>0.371872</td>
      <td>0</td>
      <td>0</td>
      <td>0.118118</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16748</th>
      <td>0.651151</td>
      <td>0.371872</td>
      <td>0</td>
      <td>0</td>
      <td>0.478478</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5899</th>
      <td>0.651151</td>
      <td>0.371872</td>
      <td>0</td>
      <td>0</td>
      <td>0.605606</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22327</th>
      <td>0.353353</td>
      <td>0.701201</td>
      <td>0</td>
      <td>0</td>
      <td>0.036036</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Just confirming that all of our separate dataframes have the same amount of rows, and the
# matching dataframes between the training and testing sets have the same amount of columns
# so that we don't run into issues when we try to add them together and use them for our model.

print('Count Vectorizer:')
print('Training Data:', X_train.shape, train_selftext_cv_df.shape, train_title_cv_df.shape)
print('Testing Data:', X_test.shape, test_selftext_cv_df.shape, test_title_cv_df.shape)
print('\n')
print('Tfidf Vectorizer with Truncated SVD:')
print('Training Data:', X_train.shape, X_train_selftext_tfidf_df.shape, X_train_title_tfidf_df.shape)
print('Testing Data:', X_test.shape, X_test_selftext_tfidf_df.shape, X_test_title_tfidf_df.shape)
```

    Count Vectorizer:
    Training Data: (14566, 23) (14566, 3323) (14566, 981)
    Testing Data: (3642, 23) (3642, 3323) (3642, 981)
    
    
    Tfidf Vectorizer with Truncated SVD:
    Training Data: (14566, 23) (14566, 2500) (14566, 1000)
    Testing Data: (3642, 23) (3642, 2500) (3642, 1000)



```python
# Merging our three dataframes for our training data back into one dataframe that we will use
# for modeling. Using a left merge on the last column of each of the original dataframes so
# that we do not drop any columns.

modeling_cv_X_train = pd.merge(X_train, train_selftext_cv_df, left_on=X_train.columns[-1], 
                            right_index=True, how='left', sort=False)

modeling_cv_X_train = pd.merge(modeling_cv_X_train, train_title_cv_df, 
                            left_on=modeling_cv_X_train.columns[-1], right_index=True, 
                            how='left', sort=False)
```


```python
# Now doing the same merging of dataframes for our testing data.

modeling_cv_X_test = pd.merge(X_test, test_selftext_cv_df, left_on=X_test.columns[-1], 
                           right_index=True, how='left', sort=False)

modeling_cv_X_test = pd.merge(modeling_cv_X_test, test_title_cv_df, 
                           left_on=modeling_cv_X_test.columns[-1], right_index=True, 
                           how='left', sort=False)

print(modeling_cv_X_train.shape, '\n', modeling_cv_X_test.shape)
```

    (14566, 4318) 
     (3642, 4318)



```python
modeling_tfidf_X_train = pd.concat([X_train, X_train_title_tfidf_df], 
                                  axis=1)
modeling_tfidf_X_train = pd.concat([modeling_tfidf_X_train, 
                                  X_train_selftext_tfidf_df], axis=1)
```


```python
modeling_tfidf_X_test = pd.concat([X_test, X_test_title_tfidf_df], 
                                  axis=1)
modeling_tfidf_X_test = pd.concat([modeling_tfidf_X_test, 
                                  X_test_selftext_tfidf_df], axis=1)
```


```python
print(modeling_tfidf_X_train.shape, modeling_tfidf_X_test.shape)
```

    (14566, 3523) (3642, 3523)


# Modeling


```python
fig, ax = plt.subplots(figsize=(20,15))

plt.bar(np.arange(len(df.subreddit.value_counts())), 
        df.subreddit.value_counts(), tick_label=df.subreddit.value_counts().values)

plt.xticks(np.arange(len(df.subreddit.value_counts())), 
           df['subreddit'].value_counts().index, rotation=90, fontsize=22)

rects = ax.patches

labels = df.subreddit.value_counts()

for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', 
            fontsize=15)

plt.yticks(fontsize=20)

plt.xlabel('Subreddits', fontsize=40)
plt.ylabel('Number of Occurences in Dataset', fontsize=40)
plt.title('Number of Each Subreddit in Dataset', fontsize=50)

plt.show()
```


![png](/images/Submission_files/Submission_88_0.png)



```python
pie_df = pd.DataFrame([df.subreddit.value_counts()[0], 
                       df.subreddit.value_counts()[1],
                       df.subreddit.value_counts()[2],
                       df.subreddit.value_counts()[3:].sum()], 
                       index=[df['subreddit'].value_counts().index[0],
                       df['subreddit'].value_counts().index[1],
                       df['subreddit'].value_counts().index[2],
                       'bottom '+str(len(df.subreddit.value_counts()[3:]))+' subreddits'])

pie_df.plot(kind='pie', subplots=True, figsize=(12,12), fontsize=26,
           legend=False, title='Subreddit Distribution in Dataset')

# Cannot seem to be able to increase the size of the title in pandas plots

# I know it is generally bad practice to use pie charts, but I just wanted to show the rough 
# distribution of subreddits in the dataset. I believe that the piechart gets this across very
# well at a quick glance.
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x1a857a0be0>], dtype=object)




![png](/images/Submission_files/Submission_89_1.png)


As stated before, we will be trying to predict the subreddit of each post in our dataset. As we can see in the graphs above, the vast majority of entries in the dataset belong to just a few different subreddits. With techsupport having a little more than half, and a little bit less than a quarter being shared between learnprogramming and learnpython, that leaves only about a quarter of all entries split between the remaining 32 subreddits.

This could cause issues with our model, as I believe it will be very unlikely to predict any of the subreddits that don't even have a visible bar on the barchart. There are just too few entries to train on. I also believe that this will be balanced out with our model predicting much too many techsupport values, and also probably learnprogramming and learnpython. 


```python
# Calculating the baseline accuracy of our model - the percentage of correct classifications
# that we want to beat, in order for our model to be better than just guessing that every
# entry's subreddit is techsupport.

baseline = df['subreddit'].value_counts()[0] / df['subreddit'].value_counts().sum()
print(baseline)
```

    0.525538224956


Our baseline accuracy for our model is about 52.6%.


```python
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
```

### Using Count Vectorizer Data


```python
randomforest = RandomForestClassifier(n_estimators=50, max_features=None)

randomforest.fit(modeling_cv_X_train, y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=None, max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)




```python
randomforest.score(modeling_cv_X_train, y_train)
```




    0.97061650418783474




```python
cross_val_score(randomforest, modeling_cv_X_test, y_test, cv=5, n_jobs=-1).mean()
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)





    0.47127791213120329




```python
lr = LogisticRegression(penalty='l1', n_jobs=-1)

lr.fit(modeling_cv_X_train, y_train)
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
              penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
lr.score(modeling_cv_X_train, y_train)
```




    0.562130989976658




```python
cross_val_score(lr, modeling_cv_X_test, y_test, cv=5).mean()
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/logistic.py:1228: UserWarning: 'n_jobs' > 1 does not have any effect when 'solver' is set to 'liblinear'. Got 'n_jobs' = -1.
      " = {}.".format(self.n_jobs))





    0.55303138702639565



Our CountVectorizer data did not perform great. The base LogisticRegression model was able to beat the baseline, but only by ~3%. I will now try the TfidfVectorizer with Truncated SVD data to see if it performs better.

### Using Tfidf Vectorizer Truncated SVD Data


```python
lr = LogisticRegression()

lr.fit(modeling_tfidf_X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)




```python
lr.score(modeling_tfidf_X_train, y_train)
```




    0.84786489084168615




```python
cross_val_score(lr, modeling_tfidf_X_test, y_test, cv=5).mean()
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)





    0.72056287803503871




```python
lr_lasso = LogisticRegression(penalty='l1', solver='saga')

lr_lasso.fit(modeling_tfidf_X_train, y_train)
lr_lasso.score(modeling_tfidf_X_train, y_train)
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/linear_model/sag.py:326: ConvergenceWarning: The max_iter was reached which means the coef_ did not converge
      "the coef_ did not converge", ConvergenceWarning)





    0.80955650144171354




```python
cross_val_score(lr_lasso, modeling_tfidf_X_test, y_test, cv=5).mean()
```

Right off the bat the Tfidf with TSVD data performs much better, with a 72% testing set accuracy as opposed to the 52.6% baseline. I will now attempt to tune this model and see how high we can get it.


```python
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

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 2 members, which is too few. The minimum number of members in any class cannot be less than n_splits=3.
      % (min_groups, self.n_splits)), Warning)


    {'C': 5, 'multi_class': 'multinomial', 'solver': 'sag'}
    0.790608265825



```python
mn_lr_C5 = LogisticRegression(solver='sag', multi_class='multinomial',
                             C=5)

mn_lr_C5.fit(modeling_tfidf_X_train, y_train)
mn_lr_C5.score(modeling_tfidf_X_train, y_train)

# I am not exactly sure why the score given here is so much higher than the best score output
# by the gridsearch with the same parameters, but I am not complaining!
```




    0.97219552382260055




```python
cross_val_score(mn_lr_C5, modeling_tfidf_X_test, y_test, cv=5).mean()
```

    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/model_selection/_split.py:605: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)





    0.75240936888778776



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
              2       0.61      0.74      0.67       305
              3       0.78      0.83      0.80       155
              4       0.63      0.56      0.59        84
              5       0.61      0.48      0.54        85
              6       0.54      0.36      0.43        81
              7       0.79      0.81      0.80        54
              8       0.41      0.11      0.18        61
              9       0.58      0.38      0.46        58
             10       0.69      0.39      0.50        46
             11       0.94      0.39      0.55        44
             12       0.34      0.34      0.34        29
             13       0.73      0.38      0.50        29
             14       0.69      0.35      0.46        26
             15       0.85      0.46      0.59        24
             16       0.17      0.05      0.08        19
             17       0.67      0.14      0.24        14
             18       0.57      0.36      0.44        11
             19       1.00      0.22      0.36         9
             20       0.70      0.58      0.64        12
             21       1.00      0.50      0.67         8
             22       0.00      0.00      0.00         7
             23       1.00      0.12      0.22         8
             24       0.00      0.00      0.00         6
             25       0.00      0.00      0.00         7
             26       0.67      0.25      0.36         8
             27       0.50      0.33      0.40         3
             28       0.00      0.00      0.00         5
             29       1.00      1.00      1.00         1
             31       0.00      0.00      0.00         3
             32       0.00      0.00      0.00         0
             33       1.00      1.00      1.00         1
    
    avg / total       0.79      0.80      0.79      3642
    


    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)



```python
predicted_classes = df.subreddit.value_counts()\
                                .index[:30].append(df.subreddit.value_counts().index[31:34])
```


```python
print(classification_report(y_test, predicted, target_names=pred))
```

                      precision    recall  f1-score   support
    
         techsupport       0.94      0.99      0.96      1891
    learnprogramming       0.61      0.74      0.67       548
         learnpython       0.61      0.74      0.67       305
             gamedev       0.78      0.83      0.80       155
          web_design       0.63      0.56      0.59        84
            javahelp       0.61      0.48      0.54        85
          javascript       0.54      0.36      0.43        81
             csshelp       0.79      0.81      0.80        54
              Python       0.41      0.11      0.18        61
      iOSProgramming       0.58      0.38      0.46        58
               linux       0.69      0.39      0.50        46
         engineering       0.94      0.39      0.55        44
               swift       0.34      0.34      0.34        29
     computerscience       0.73      0.38      0.50        29
              django       0.69      0.35      0.46        26
                 PHP       0.85      0.46      0.59        24
                 css       0.17      0.05      0.08        19
                java       0.67      0.14      0.24        14
                HTML       0.57      0.36      0.44        11
                ruby       1.00      0.22      0.36         9
               flask       0.70      0.58      0.64        12
             compsci       1.00      0.50      0.67         8
          technology       0.00      0.00      0.00         7
                 cpp       1.00      0.12      0.22         8
               html5       0.00      0.00      0.00         6
              pygame       0.00      0.00      0.00         7
              jquery       0.67      0.25      0.36         8
                perl       0.50      0.33      0.40         3
                lisp       0.00      0.00      0.00         5
     dailyprogrammer       1.00      1.00      1.00         1
             IPython       0.00      0.00      0.00         3
    inventwithpython       0.00      0.00      0.00         0
              netsec       1.00      1.00      1.00         1
    
         avg / total       0.79      0.80      0.79      3642
    


    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/stephenhockey/anaconda3/lib/python3.6/site-packages/sklearn/metrics/classification.py:1137: UndefinedMetricWarning: Recall and F-score are ill-defined and being set to 0.0 in labels with no true samples.
      'recall', 'true', average, warn_for)



```python
con_mat = confusion_matrix(y_test, predicted)

con_mat_index = []
con_mat_columns = []
for subreddit in predicted_classes:
    con_mat_index.append('is_' + subreddit)
    con_mat_columns.append('predicted_' + subreddit)
    
confusion = pd.DataFrame(con_mat, index=con_mat_index, columns=con_mat_columns)confusion.iloc[:, :20]
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
      <th>predicted_techsupport</th>
      <th>predicted_learnprogramming</th>
      <th>predicted_learnpython</th>
      <th>predicted_gamedev</th>
      <th>predicted_web_design</th>
      <th>predicted_javahelp</th>
      <th>predicted_javascript</th>
      <th>predicted_csshelp</th>
      <th>predicted_Python</th>
      <th>predicted_iOSProgramming</th>
      <th>predicted_linux</th>
      <th>predicted_engineering</th>
      <th>predicted_swift</th>
      <th>predicted_computerscience</th>
      <th>predicted_django</th>
      <th>predicted_PHP</th>
      <th>predicted_css</th>
      <th>predicted_java</th>
      <th>predicted_HTML</th>
      <th>predicted_ruby</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_techsupport</th>
      <td>1865</td>
      <td>13</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_learnprogramming</th>
      <td>24</td>
      <td>405</td>
      <td>56</td>
      <td>10</td>
      <td>10</td>
      <td>17</td>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_learnpython</th>
      <td>11</td>
      <td>51</td>
      <td>226</td>
      <td>4</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_gamedev</th>
      <td>4</td>
      <td>16</td>
      <td>3</td>
      <td>128</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_web_design</th>
      <td>9</td>
      <td>17</td>
      <td>2</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_javahelp</th>
      <td>7</td>
      <td>27</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>41</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_javascript</th>
      <td>7</td>
      <td>28</td>
      <td>7</td>
      <td>4</td>
      <td>3</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_csshelp</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_Python</th>
      <td>2</td>
      <td>13</td>
      <td>33</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_iOSProgramming</th>
      <td>4</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_linux</th>
      <td>15</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_engineering</th>
      <td>7</td>
      <td>13</td>
      <td>2</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_swift</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_computerscience</th>
      <td>4</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_django</th>
      <td>2</td>
      <td>2</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_PHP</th>
      <td>1</td>
      <td>5</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_css</th>
      <td>3</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_java</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_HTML</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_ruby</th>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>is_flask</th>
      <td>0</td>
      <td>2</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_compsci</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_technology</th>
      <td>5</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_cpp</th>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_html5</th>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_pygame</th>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_jquery</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_perl</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_lisp</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_dailyprogrammer</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_IPython</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_inventwithpython</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_netsec</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
confusion.iloc[:, 20:]
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
      <th>predicted_flask</th>
      <th>predicted_compsci</th>
      <th>predicted_technology</th>
      <th>predicted_cpp</th>
      <th>predicted_html5</th>
      <th>predicted_pygame</th>
      <th>predicted_jquery</th>
      <th>predicted_perl</th>
      <th>predicted_lisp</th>
      <th>predicted_dailyprogrammer</th>
      <th>predicted_IPython</th>
      <th>predicted_inventwithpython</th>
      <th>predicted_netsec</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>is_techsupport</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_learnprogramming</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_learnpython</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_gamedev</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_web_design</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_javahelp</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_javascript</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_csshelp</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_Python</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_iOSProgramming</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_linux</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_engineering</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_swift</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_computerscience</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_django</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_PHP</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_css</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_java</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_HTML</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_ruby</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_flask</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_compsci</th>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_technology</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_cpp</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_html5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_pygame</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_jquery</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_perl</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_lisp</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_dailyprogrammer</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_IPython</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_inventwithpython</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>is_netsec</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# the two classifications which did not appear in the test set or get predicted by the model
# and are therefore omitted from the confusion matrix and classification report
print(df.subreddit.value_counts().index[30], df.subreddit.value_counts().index[34])
```

    programmer pystats



```python
# Just confirming that index 30 (programmer) and 34 (pystats) from above did not appear in 
# y_test. 32 (inventwithpython) also did not appear in testing set, but had one predicted value
# with was actually in the subreddit learnwithpython
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
    21       8
    26       8
    23       8
    22       7
    25       7
    24       6
    28       5
    27       3
    31       3
    29       1
    33       1
    Name: subreddit, dtype: int64



Since we only have 33 different labels out of the original 35 classifications, we must have lost two of them somewhere along the way. I would assume that the very small amount of entries in several of the different classifications made it so that those classes did not appear at all in the testing data and did not get predicted at all either, therefore omitting them from the classification report. This is because we can see one class, inventwithpython, has 0 values in the testing set meaning it must have been predicted at least once by my model in order for it to show up. 

So, as predicted, the multiple classifications with very few entries caused some minor problems with my model, though not only in the way I originally anticipated. There are three classifications (inventwithpython, programmer and pystats) that did not have a single entry in my testing set as there were too few values and the test_size of 20% did not happen to fall on any of them.

Also as predicted, my model did extremely well at predicting whether or not the post was in techsupport, but did noticeably worse for every other subreddit. Unfortunately, my model was not as good as I expected it would be for the other two sizable classes, learnprogramming and learnpython, but it did better than expected in the fourth most populous class gamedev.

It's quite hard to read a confusion matrix this big as the jupyter notebook cuts out columns when you have more than 20. However most of the important information is contained in the first one anyways. The areas where my most inaccurate from a scan of the confusion matrix are
the classifications with fewer entries that also closely relate in topic to more populous
classifications. For example: the model only correctly predicted 7 of the Python subreddit, but misclassified 33 of them as learnpython and 13 as learnprogramming, much more popular
subreddits. css also only had one true positive, but three misclassifications in both 
techsupport and learnprogramming.

# Next Steps

Because I completely got rid of the deleted and removed posts before modeling in order to more accurately predict the subreddit of posts which actual 'selftext' values and rely less on the natural language processed 'title' columns, I would like to make a separate model for the entries I negated. It would be interesting to see how accurately I could predict the subreddit of a deleted or removed post using just the processed 'title' columns compared to the accuracy of the model with everything included.

I would also like to see if I could add an ROC curve to my model evaluation piece. I could not seem to get it to work at all for my multi-class classification as I had only done it before with binary classification problems. I'm assuming there is an easy fix to the problems I had but could not figure it out in my limited time I had left.
