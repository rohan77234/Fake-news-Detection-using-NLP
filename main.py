import pandas as pd
import nltk

nltk.download("punkt")

fake = pd.read_csv("Fake-211023-185413.csv")
not_fake = pd.read_csv("True-211023-185340.csv")

print(fake.info())
print()
print()
print(not_fake.info())
print()
print()
print(fake.head(5))
print()
print()
print(not_fake.head(5))

print(fake.subject.value_counts())

fake['target'] = 0
not_fake['target'] = 1

data = pd.concat([fake, not_fake ], axis=0)
print(data.head(5))
print(data.tail(5))

data = data.reset_index(drop=True)

data = data.drop(['subject', 'date', 'title'], axis=1)
print(data.columns)

#### tokenise
from nltk.tokenize import word_tokenize

data['text'] = data['text'].apply(word_tokenize)
print(data.head(10))
####stemming
from nltk.stem.snowball import SnowballStemmer
porter = SnowballStemmer("english")

def stem_it(text):
    return [porter.stem(word) for word in text]

data['text'] = data['text'].apply(stem_it)
print(data.head(10))

# from nltk.corpus import stopwords
def stop_it(t):
    dt = [word for word in t if len(word)>2]
    return dt

data['text'] = data['text'].apply(stop_it)
print(data.head(10))

data['text'] = data['text'].apply(' '.join)
#### splitting

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(data['text'],data['target'], test_size=0.25)
print(X_train.head())
print()
print(y_train.head())

#### Vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer
mytfidf = TfidfVectorizer(max_df=0.7)

tfidf_train = mytfidf.fit_transform(X_train)
tfidf_test = mytfidf.transform(X_test)
print(tfidf_test)

##### ligistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


model_1 = LogisticRegression(max_iter=900)
model_1.fit(tfidf_train, y_train)
pred_1 = model_1.predict(tfidf_test)
cr1 = accuracy_score(y_test, pred_1)
print(cr1*100)


#### Passive Aggressive Classifier
from sklearn.linear_model import PassiveAggressiveClassifier

model1 = PassiveAggressiveClassifier(max_iter=50)
model1.fit(tfidf_train, y_train)

y_pred = model1.predict(tfidf_test)
accscore = accuracy_score(y_test, y_pred)
print('The accuracy of prediction is ', accscore*100)
