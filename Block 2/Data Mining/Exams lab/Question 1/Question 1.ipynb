{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Question 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "import random\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>title</th>\n",
       "      <th>label</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15 Highly Important Questions About Adulthood,...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>250 Nuns Just Cycled All The Way From Kathmand...</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Australian comedians \"could have been shot\" du...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Lycos launches screensaver to increase spammer...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Fußball-Bundesliga 2008–09: Goalkeeper Butt si...</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                               title  label\n",
       "0  15 Highly Important Questions About Adulthood,...      1\n",
       "1  250 Nuns Just Cycled All The Way From Kathmand...      1\n",
       "2  Australian comedians \"could have been shot\" du...      0\n",
       "3  Lycos launches screensaver to increase spammer...      0\n",
       "4  Fußball-Bundesliga 2008–09: Goalkeeper Butt si...      0"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Firstly we are going to train our model through the url from the manual\n",
    "import pandas as pd\n",
    "\n",
    "DATASET_URL = 'https://gist.githubusercontent.com/amitness/0a2ddbcb61c34eab04bad5a17fd8c86b/raw/66ad13dfac4bd1201e09726677dd8ba8048bb8af/clickbait.csv'\n",
    "data = pd.read_csv(DATASET_URL)\n",
    "data.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train the model  and split it 80/20 ratio train set/test set\n",
    "X = list(data.title.values)\n",
    "y = list(data.label.values)# the labels we want to predict --> Y\n",
    "labels = ['not clickbait', 'clickbait']\n",
    "\n",
    "X_train_str, X_test_str, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "#create the vector of occurance, for every word\n",
    "cv = CountVectorizer() # this initializes the CountVectorizer \n",
    "\n",
    "cv.fit(X_train_str) # create the vocabulary\n",
    "\n",
    "X_train = cv.transform(X_train_str)\n",
    "X_test = cv.transform(X_test_str)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "#print(X_train.toarray()[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>00</th>\n",
       "      <th>000</th>\n",
       "      <th>000th</th>\n",
       "      <th>00s</th>\n",
       "      <th>04</th>\n",
       "      <th>05</th>\n",
       "      <th>08</th>\n",
       "      <th>08m</th>\n",
       "      <th>09</th>\n",
       "      <th>10</th>\n",
       "      <th>...</th>\n",
       "      <th>zuckerberg</th>\n",
       "      <th>zuma</th>\n",
       "      <th>zurawski</th>\n",
       "      <th>zurich</th>\n",
       "      <th>złoty</th>\n",
       "      <th>íngrid</th>\n",
       "      <th>íslands</th>\n",
       "      <th>île</th>\n",
       "      <th>śrī</th>\n",
       "      <th>šibenik</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 20660 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   00  000  000th  00s  04  05  08  08m  09  10  ...  zuckerberg  zuma  \\\n",
       "0   0    0      0    0   0   0   0    0   0   0  ...           0     0   \n",
       "1   0    0      0    0   0   0   0    0   0   0  ...           0     0   \n",
       "2   0    0      0    0   0   0   0    0   0   0  ...           0     0   \n",
       "3   0    0      0    0   0   0   0    0   0   0  ...           0     0   \n",
       "4   0    0      0    0   0   0   0    0   0   0  ...           0     0   \n",
       "\n",
       "   zurawski  zurich  złoty  íngrid  íslands  île  śrī  šibenik  \n",
       "0         0       0      0       0        0    0    0        0  \n",
       "1         0       0      0       0        0    0    0        0  \n",
       "2         0       0      0       0        0    0    0        0  \n",
       "3         0       0      0       0        0    0    0        0  \n",
       "4         0       0      0       0        0    0    0        0  \n",
       "\n",
       "[5 rows x 20660 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#i wanted to represent the vocbulary as a dataframe to visualize the result\n",
    "vocabulary = cv.get_feature_names()\n",
    "vectorized_texts = pd.DataFrame(X_train.toarray(), columns=vocabulary)\n",
    "vectorized_texts.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,\n",
       "                   intercept_scaling=1, l1_ratio=None, max_iter=1000,\n",
       "                   multi_class='auto', n_jobs=None, penalty='l2',\n",
       "                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,\n",
       "                   warm_start=False)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#train the model with Logistic Regression\n",
    "lr = LogisticRegression(solver='lbfgs',max_iter=1000)\n",
    "lr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#load my dataset for the exercise\n",
    "data2 = pd.read_pickle(\"framing.p\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "del data2['label'] #i know that i could just name my new column otherwise but i like the name labels and for this example the label didint do anything"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tweet_id</th>\n",
       "      <th>date</th>\n",
       "      <th>user</th>\n",
       "      <th>party</th>\n",
       "      <th>state</th>\n",
       "      <th>chamber</th>\n",
       "      <th>tweet</th>\n",
       "      <th>news_mention</th>\n",
       "      <th>url_reference</th>\n",
       "      <th>netloc</th>\n",
       "      <th>title</th>\n",
       "      <th>description</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1325914751495499776</td>\n",
       "      <td>2020-11-09 21:34:45</td>\n",
       "      <td>SenShelby</td>\n",
       "      <td>R</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>ICYMI – @BusinessInsider declared #Huntsville ...</td>\n",
       "      <td>businessinsider</td>\n",
       "      <td>https://www.businessinsider.com/personal-finan...</td>\n",
       "      <td>www.businessinsider.com</td>\n",
       "      <td>The 10 best US cities to move to if you want t...</td>\n",
       "      <td>The best US cities to move to if you want to s...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1294021087118987264</td>\n",
       "      <td>2020-08-13 21:20:43</td>\n",
       "      <td>SenShelby</td>\n",
       "      <td>R</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>Great news! Today @mazda_toyota announced an a...</td>\n",
       "      <td></td>\n",
       "      <td>https://pressroom.toyota.com/mazda-and-toyota-...</td>\n",
       "      <td>pressroom.toyota.com</td>\n",
       "      <td>Mazda and Toyota Further Commitment to U.S. Ma...</td>\n",
       "      <td>HUNTSVILLE, Ala., (Aug. 13, 2020) – Today, Maz...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1323340848130609156</td>\n",
       "      <td>2020-11-02 19:06:59</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>He’s already quitting on the folks of Alabama ...</td>\n",
       "      <td></td>\n",
       "      <td>https://apnews.com/article/c73f0dfe8008ebaf85e...</td>\n",
       "      <td>apnews.com</td>\n",
       "      <td>Tuberville, Jones fight for Senate seat in Ala...</td>\n",
       "      <td>GARDENDALE, Ala. (AP) — U.S. Sen. Doug Jones, ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1323004075831709698</td>\n",
       "      <td>2020-11-01 20:48:46</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>I know you guys are getting bombarded with fun...</td>\n",
       "      <td></td>\n",
       "      <td>https://secure.actblue.com/donate/djfs-close?r...</td>\n",
       "      <td>secure.actblue.com</td>\n",
       "      <td>I just gave!</td>\n",
       "      <td>Join us! Contribute today.</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1322567531320717314</td>\n",
       "      <td>2020-10-31 15:54:06</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>Well looky here folks, his own players don’t t...</td>\n",
       "      <td></td>\n",
       "      <td>https://slate.com/culture/2020/10/tommy-tuberv...</td>\n",
       "      <td>slate.com</td>\n",
       "      <td>What Tommy Tuberville’s Former Auburn Players ...</td>\n",
       "      <td>\"All I could think is, why?\"</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              tweet_id                date       user party    state  chamber  \\\n",
       "0  1325914751495499776 2020-11-09 21:34:45  SenShelby     R  Alabama  Senator   \n",
       "1  1294021087118987264 2020-08-13 21:20:43  SenShelby     R  Alabama  Senator   \n",
       "2  1323340848130609156 2020-11-02 19:06:59  DougJones     D  Alabama  Senator   \n",
       "3  1323004075831709698 2020-11-01 20:48:46  DougJones     D  Alabama  Senator   \n",
       "4  1322567531320717314 2020-10-31 15:54:06  DougJones     D  Alabama  Senator   \n",
       "\n",
       "                                               tweet     news_mention  \\\n",
       "0  ICYMI – @BusinessInsider declared #Huntsville ...  businessinsider   \n",
       "1  Great news! Today @mazda_toyota announced an a...                    \n",
       "2  He’s already quitting on the folks of Alabama ...                    \n",
       "3  I know you guys are getting bombarded with fun...                    \n",
       "4  Well looky here folks, his own players don’t t...                    \n",
       "\n",
       "                                       url_reference                   netloc  \\\n",
       "0  https://www.businessinsider.com/personal-finan...  www.businessinsider.com   \n",
       "1  https://pressroom.toyota.com/mazda-and-toyota-...     pressroom.toyota.com   \n",
       "2  https://apnews.com/article/c73f0dfe8008ebaf85e...               apnews.com   \n",
       "3  https://secure.actblue.com/donate/djfs-close?r...       secure.actblue.com   \n",
       "4  https://slate.com/culture/2020/10/tommy-tuberv...                slate.com   \n",
       "\n",
       "                                               title  \\\n",
       "0  The 10 best US cities to move to if you want t...   \n",
       "1  Mazda and Toyota Further Commitment to U.S. Ma...   \n",
       "2  Tuberville, Jones fight for Senate seat in Ala...   \n",
       "3                                       I just gave!   \n",
       "4  What Tommy Tuberville’s Former Auburn Players ...   \n",
       "\n",
       "                                         description  \n",
       "0  The best US cities to move to if you want to s...  \n",
       "1  HUNTSVILLE, Ala., (Aug. 13, 2020) – Today, Maz...  \n",
       "2  GARDENDALE, Ala. (AP) — U.S. Sen. Doug Jones, ...  \n",
       "3                         Join us! Contribute today.  \n",
       "4                       \"All I could think is, why?\"  "
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data2.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "new_example =list(data2['title'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "#train my new dataset to the existing model thati created before and add the predictions to a constant\n",
    "new_example_bow = cv.transform(new_example)# icould use more examples like tf-idf or Bert my choice was \n",
    "predictions = lr.predict(new_example_bow)\n",
    "a = []\n",
    "for i, prediction in enumerate(predictions):   # i use that for loop so i dont have to rename tha labes from 0 or 1\n",
    "    \"{}: {}\".format(new_example[i], labels[prediction])\n",
    "    a.append(labels[prediction])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "data2['labels'] = a"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>tweet_id</th>\n",
       "      <th>date</th>\n",
       "      <th>user</th>\n",
       "      <th>party</th>\n",
       "      <th>state</th>\n",
       "      <th>chamber</th>\n",
       "      <th>tweet</th>\n",
       "      <th>news_mention</th>\n",
       "      <th>url_reference</th>\n",
       "      <th>netloc</th>\n",
       "      <th>title</th>\n",
       "      <th>description</th>\n",
       "      <th>labels</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1325914751495499776</td>\n",
       "      <td>2020-11-09 21:34:45</td>\n",
       "      <td>SenShelby</td>\n",
       "      <td>R</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>ICYMI – @BusinessInsider declared #Huntsville ...</td>\n",
       "      <td>businessinsider</td>\n",
       "      <td>https://www.businessinsider.com/personal-finan...</td>\n",
       "      <td>www.businessinsider.com</td>\n",
       "      <td>The 10 best US cities to move to if you want t...</td>\n",
       "      <td>The best US cities to move to if you want to s...</td>\n",
       "      <td>clickbait</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1294021087118987264</td>\n",
       "      <td>2020-08-13 21:20:43</td>\n",
       "      <td>SenShelby</td>\n",
       "      <td>R</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>Great news! Today @mazda_toyota announced an a...</td>\n",
       "      <td></td>\n",
       "      <td>https://pressroom.toyota.com/mazda-and-toyota-...</td>\n",
       "      <td>pressroom.toyota.com</td>\n",
       "      <td>Mazda and Toyota Further Commitment to U.S. Ma...</td>\n",
       "      <td>HUNTSVILLE, Ala., (Aug. 13, 2020) – Today, Maz...</td>\n",
       "      <td>not clickbait</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1323340848130609156</td>\n",
       "      <td>2020-11-02 19:06:59</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>He’s already quitting on the folks of Alabama ...</td>\n",
       "      <td></td>\n",
       "      <td>https://apnews.com/article/c73f0dfe8008ebaf85e...</td>\n",
       "      <td>apnews.com</td>\n",
       "      <td>Tuberville, Jones fight for Senate seat in Ala...</td>\n",
       "      <td>GARDENDALE, Ala. (AP) — U.S. Sen. Doug Jones, ...</td>\n",
       "      <td>not clickbait</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1323004075831709698</td>\n",
       "      <td>2020-11-01 20:48:46</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>I know you guys are getting bombarded with fun...</td>\n",
       "      <td></td>\n",
       "      <td>https://secure.actblue.com/donate/djfs-close?r...</td>\n",
       "      <td>secure.actblue.com</td>\n",
       "      <td>I just gave!</td>\n",
       "      <td>Join us! Contribute today.</td>\n",
       "      <td>clickbait</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1322567531320717314</td>\n",
       "      <td>2020-10-31 15:54:06</td>\n",
       "      <td>DougJones</td>\n",
       "      <td>D</td>\n",
       "      <td>Alabama</td>\n",
       "      <td>Senator</td>\n",
       "      <td>Well looky here folks, his own players don’t t...</td>\n",
       "      <td></td>\n",
       "      <td>https://slate.com/culture/2020/10/tommy-tuberv...</td>\n",
       "      <td>slate.com</td>\n",
       "      <td>What Tommy Tuberville’s Former Auburn Players ...</td>\n",
       "      <td>\"All I could think is, why?\"</td>\n",
       "      <td>not clickbait</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              tweet_id                date       user party    state  chamber  \\\n",
       "0  1325914751495499776 2020-11-09 21:34:45  SenShelby     R  Alabama  Senator   \n",
       "1  1294021087118987264 2020-08-13 21:20:43  SenShelby     R  Alabama  Senator   \n",
       "2  1323340848130609156 2020-11-02 19:06:59  DougJones     D  Alabama  Senator   \n",
       "3  1323004075831709698 2020-11-01 20:48:46  DougJones     D  Alabama  Senator   \n",
       "4  1322567531320717314 2020-10-31 15:54:06  DougJones     D  Alabama  Senator   \n",
       "\n",
       "                                               tweet     news_mention  \\\n",
       "0  ICYMI – @BusinessInsider declared #Huntsville ...  businessinsider   \n",
       "1  Great news! Today @mazda_toyota announced an a...                    \n",
       "2  He’s already quitting on the folks of Alabama ...                    \n",
       "3  I know you guys are getting bombarded with fun...                    \n",
       "4  Well looky here folks, his own players don’t t...                    \n",
       "\n",
       "                                       url_reference                   netloc  \\\n",
       "0  https://www.businessinsider.com/personal-finan...  www.businessinsider.com   \n",
       "1  https://pressroom.toyota.com/mazda-and-toyota-...     pressroom.toyota.com   \n",
       "2  https://apnews.com/article/c73f0dfe8008ebaf85e...               apnews.com   \n",
       "3  https://secure.actblue.com/donate/djfs-close?r...       secure.actblue.com   \n",
       "4  https://slate.com/culture/2020/10/tommy-tuberv...                slate.com   \n",
       "\n",
       "                                               title  \\\n",
       "0  The 10 best US cities to move to if you want t...   \n",
       "1  Mazda and Toyota Further Commitment to U.S. Ma...   \n",
       "2  Tuberville, Jones fight for Senate seat in Ala...   \n",
       "3                                       I just gave!   \n",
       "4  What Tommy Tuberville’s Former Auburn Players ...   \n",
       "\n",
       "                                         description         labels  \n",
       "0  The best US cities to move to if you want to s...      clickbait  \n",
       "1  HUNTSVILLE, Ala., (Aug. 13, 2020) – Today, Maz...  not clickbait  \n",
       "2  GARDENDALE, Ala. (AP) — U.S. Sen. Doug Jones, ...  not clickbait  \n",
       "3                         Join us! Contribute today.      clickbait  \n",
       "4                       \"All I could think is, why?\"  not clickbait  "
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data2.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Now we will count how many times democrats and republicans use clickbait titles and compare them"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Republicans are : 7771\n",
      "Democrats are : 15564\n"
     ]
    }
   ],
   "source": [
    "#we create 2 new attributes so its easy to count the numbers of each other party labels\n",
    "data2_rep = data2[data2['party']=='R']\n",
    "data2_dem = data2[data2['party']=='D']\n",
    "print(\"Republicans are :\",len(data2_rep))\n",
    "print(\"Democrats are :\",len(data2_dem))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1058\n"
     ]
    }
   ],
   "source": [
    "#thats how many times repablicans used clickbait labales\n",
    "rep_times = data2_rep[data2_rep.labels == 'clickbait']['labels'].count()\n",
    "print(rep_times)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "2830\n"
     ]
    }
   ],
   "source": [
    "#thats how many times democrats used clickbait labales\n",
    "dem_times = data2_dem[data2_dem.labels == 'clickbait']['labels'].count()\n",
    "print(dem_times)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "And if we want the proportion of both partys:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.1361472140007721\n"
     ]
    }
   ],
   "source": [
    "#now for the repablicans\n",
    "proportion_rep_click = rep_times/len(data2_rep)\n",
    "print(proportion_rep_click)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.18182986378822924\n"
     ]
    }
   ],
   "source": [
    "#and now for the democrats\n",
    "proportion_dem_click = dem_times/len(data2_dem)\n",
    "print(proportion_dem_click)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "So in conclusion the democrats refer to an article with a clickbait title 2830 times while the republicans only 1058 even the difference in the sample is big democrats are way more in this sample the proportion is still bigger for democrats rather than republicans.\n",
    "After viewing the clickbaited onces i saw that most of them where not making any sence and they where either rubbish or advertisments,also they seem to have some actual \"conversation\" like \"i know that you boomed with advertisments\" so they most are in 1st person as far as i reviwed them."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
