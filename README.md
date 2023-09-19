# Trump Tweets: # of Likes and Word Count

## Question

What is the effect of the number of words in a Trump tweet on the number of favorites/likes received on the post?

### E[# likes / tweet | # words / tweet]

In this project, we aim to study the dataset of Trump tweets found from [the trump archive](https://www.thetrumparchive.com/faq).

## Approach

### Confounders

We consider that the year of the tweets, Tweeting behavior of the tweeter (Trump), including tweet length and content, and the political climate may be different year by year, so we attempt to normalize this by classifying our data into different groups by year, and remove the confounder of the the current time period context. However, this is more of a high level attempt to address this confounder, and it may change even more minutely at the week or month level (depends on the events that go on at that time). The actual sentiment and content of the tweet itself is also a huge confounder that requires machine learning for sentiment analysis.

### Load in Data

We import the above CSV dataset of Trump tweets into Google Drive, and then mount the data like so into our Jupyter Notebook. We must manually grant access to our Google account.

```python
from google.colab import drive

drive.mount("/content/gdrive")
full_df = pd.read_csv("gdrive/My Drive/full_raw_data.csv")
```

### Analysis

We clean the dataframe accordingly and also derive the correlation between word count and number of likes for every tweet,filtering based on year between years 2017-2020.

```python
#read in trumptweets.csv
import numpy as np
import matplotlib.pyplot as plt

import string
from scipy.stats import pearsonr

years = [2017, 2018, 2019, 2020]

for year in years:
  yearSt = str(year)
  #delete rows where isRetweet is true
  df = fulldf[fulldf.isRetweet != 't']
  #create new row value that contains tweet word count
  df['wordCount'] = df['text'].str.split().str.len()
  #remove instances of url to remove bias for lower word count
  df = df[~df.text.str.contains("http")]
  df = df[df.date.str.contains(yearSt)]

  df = df.sort_values(by=['favorites'])
  pd.set_option('display.max_colwidth', None)
  df.tail(5)

  plt.scatter(df['wordCount'], df['favorites'])

  m, b = np.polyfit(df['wordCount'], df['favorites'], 1)
  plt.plot(df['wordCount'], m*df['wordCount'] + b, color='red')
  plt.xlabel("Word Count of Tweet")
  plt.ylabel("Number of Likes")
  plt.title("Trump Tweets Correlations in " + yearSt)
  plt.show()
  corr, _ = pearsonr(df['favorites'], df['wordCount'])
  print('Pearsons correlation: %.3f' % corr)
```

### 2017
![Head](/assets/2017.png)

```python
Pearsons correlation: 0.139
```

### 2018
![Head](/assets/2018.png)

```python
Pearsons correlation: 0.040
```

### 2019
![Head](/assets/2019.png)

```python
Pearsons correlation: -0.027
```

### 2019
![Head](/assets/2019.png)

```python
Pearsons correlation: -0.281
```

#### Conclusions

From the plots above of observed word count of Trump’s Tweets throughout the years of his
presidency, and the magnitude of the public’s reaction measured by the number of likes and
interactions, we are able to derive information from the resultant correlation. The first observation
we make is of the Pearson correlation coefficient (r-value) between the number of likes and word
count over the years; in 2017, it starts positive at 0.139, and consistently decreases with each year
(.139 → .040 → -.027 → -.281), to the point where the correlation in the present (2020) is
essentially twice the magnitude of the initial 2017 correlation, but negative. In other words,
Trump’s shorter, more concise tweets have been receiving more attention and interactions than
his longer ones as years go by, as the largest number of likes are skewed towards a tweet with low
word count. This not only goes to show the virality of short messages on the internet, which is
tied to the general population’s shrinking attention span, but also the evolution of Trump’s
tweeting style, and the controversial messages he gets across in such little words in the context of
events at the time–that create such virality.

It appears that over the years, Trump’s habit of tweeting shorter tweets has been more common, as
the cluster of data points on the very left of the plot is much more concentrated, with higher
weight. In 2019 and 2020 especially, the cluster of tweets that are less than 10 words especially
stands out in the scatterplot. Trump’s iconic choice of a style of punctuality with low word count
is inherently linked to the syntax of his tweets, as a lower word count typically implies a more syntactically simple body of text. Not only does a higher tweet concentration of decreased word count perhaps indicate short outbursts of emotion, but it may also suggest a growing inability to form complex sentences on average. 

### Reflection

Using LLMs such as ChatGPT was helpful to get useful syntax typs, given that the schema of the dataset was known (filtered and limited to columns of number of favorites and the actual tweet involved). However, it was quite limited in terms of fully addressing confounders, especially given limited resources and time – it was able to give specific feedback on correlations between the two variables, but not able to fully extrapolate the more abstract and involved confounders and come up with targeted techniques.
I had an overall high approach in mind to take, and the LLM helped me step by step in order to reach the final destination, but could not take me further.
