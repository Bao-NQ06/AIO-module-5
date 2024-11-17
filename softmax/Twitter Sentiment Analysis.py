import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

dataset_path ='softmax\Twitter_Data.csv'
df = pd.read_csv (
dataset_path
)

df = df.dropna ()

def text_normalize ( text ) :
    # Lowercasing
    text = text.lower ()

    # Retweet old acronym "RT" removal
    text = re.sub(r'^rt [\s]+','', text )

    # Hyperlinks removal
    text = re.sub(r'https ?:\/\/.*[\ r\n]*','', text )

    # Punctuation removal
    text = re.sub (r'[^\ w\s]','', text )

    # Remove stopwords
    stop_words = set ( stopwords.words ('english') )
    words = text.split ()
    words = [ word for word in words if word not in stop_words ]
    text =''.join ( words )
    
    # Stemming
    stemmer = SnowballStemmer ('english')
    words = text.split ()
    words = [ stemmer.stem ( word ) for word in words ]
    text =''.join ( words )
    
    return text

vectorizer = TfidfVectorizer ( max_features =2000)
X = vectorizer.fit_transform (df['clean_text']).toarray ()


intercept = np.ones ((X.shape [0] , 1))
X_b = np.concatenate (( intercept , X) ,axis =1)

n_classes = df['category'].nunique ()
n_samples = df['category'].size

y = df['category'].to_numpy () + 1
y = y.astype (np.uint8 )
y_encoded = np.array ([np.zeros ( n_classes ) for _ in range ( n_samples )])
y_encoded [np.arange ( n_samples ) , y] = 1


val_size = 0.2
test_size = 0.125
random_state = 2
is_shuffle = True

X_train , X_val , y_train , y_val = train_test_split (
X_b , y_encoded ,
test_size = val_size ,
random_state = random_state ,
shuffle = is_shuffle
)

X_train , X_test , y_train , y_test = train_test_split (
X_train , y_train ,
test_size = test_size ,
random_state = random_state ,
shuffle = is_shuffle
)

def softmax (z) :
    exp_z = np.exp (z)

    return exp_z / exp_z.sum ( axis =1) [: , None ]

def compute_loss (y_hat , y) :
    n = y.size

    return ( -1 / n) * np.sum (y * np.log ( y_hat ) )

def predict (X, theta ) :
    z = np.dot (X, theta )
    y_hat = softmax (z)

    return y_hat

def compute_gradient (X, y, y_hat ) :
    n = y.size

    return np.dot(X.T, ( y_hat - y) ) / n

def update_theta (theta , gradient , lr) :
    return theta - lr * gradient

def compute_accuracy (X, y, theta ) :
    y_hat = predict (X, theta )
    acc = (np.argmax (y_hat , axis =1) == np.argmax (y, axis =1) ).mean ()
    return acc


lr = 0.1
epochs = 200
batch_size = X_train.shape [0]
n_features = X_train.shape [1]

np.random.seed ( random_state )
theta = np.random.uniform (
size =( n_features , n_classes ))

val_set_acc = compute_accuracy (X_val , y_val , theta )
test_set_acc = compute_accuracy ( X_test , y_test , theta )
print ('Evaluation on validation and test set :')
print (f'Accuracy : { val_set_acc }')
print (f'Accuracy : { test_set_acc }')