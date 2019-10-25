import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

columns = ['Age','Workclass','fnlgwt','Education','Education num','Marital Status',
           'Occupation','Relationship','Race','Sex','Capital Gain','Capital Loss',
           'Hours/Week','Native country','Income']

def primary(x):
    if x in [' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th']:
        return ' Primary'
    else:
        return x
    
def native(country):
    if country in [' United-States', ' Cuba', ' 0']:
        return 'US'
    elif country in [' England', ' Germany', ' Canada', 
                     ' Italy', ' France', ' Greece', ' Philippines']:
        return 'Western'
    elif country in [' Mexico', ' Puerto-Rico', ' Honduras', ' Jamaica', 
                     ' Columbia', ' Laos', ' Portugal', ' Haiti', 
                     ' Dominican-Republic', ' El-Salvador', ' Guatemala', ' Peru', 
                     ' Trinadad&Tobago', ' Outlying-US(Guam-USVI-etc)', ' Nicaragua',
                     ' Vietnam', ' Holand-Netherlands' ]:
        return 'Poor' # no offence
    elif country in [' India', ' Iran', ' Cambodia', ' Taiwan', 
                     ' Japan', ' Yugoslavia', ' China', ' Hong']:
        return 'Eastern'
    elif country in [' South', ' Poland', ' Ireland', ' Hungary', 
                     ' Scotland', ' Thailand', ' Ecuador']:
        return 'Poland team'
    else:
        return country


class DataModule:
    def __init__(self, normalize=True, append_one=True):
        self.normalize = normalize
        self.append_one = append_one
    
    def load(self):
        pass
    
    def fetch(self, n_tr, n_val, n_test, seed=0):
        x, y = self.load()
        
        # split data
        x_tr, x_val, y_tr, y_val = train_test_split(
            x, y, train_size=n_tr, test_size=n_val+n_test, random_state=seed)
        x_val, x_test, y_val, y_test = train_test_split(
            x_val, y_val, train_size=n_val, test_size=n_test, random_state=seed+1)
        
        # process x
        if self.normalize:
            scaler = StandardScaler()
            scaler.fit(x_tr)
            x_tr = scaler.transform(x_tr)
            x_val = scaler.transform(x_val)
            x_test = scaler.transform(x_test)
        if self.append_one:
            x_tr = np.c_[x_tr, np.ones(n_tr)]
            x_val = np.c_[x_val, np.ones(n_val)]
            x_test = np.c_[x_test, np.ones(n_test)]
        
        return (x_tr, y_tr), (x_val, y_val), (x_test, y_test)
        

class MnistModule(DataModule):
    def __init__(self, normalize=True, append_one=False):
        DataModule.__init__(self, normalize, append_one)
        from tensorflow.examples.tutorials.mnist import input_data
        self.input_data = input_data
    
    def load(self):
        mnist = self.input_data.read_data_sets('/tmp/data/', one_hot=True)
        ytr = mnist.train.labels
        xtr = mnist.train.images
        xtr1 = xtr[ytr[:, 1]>0, :]
        xtr7 = xtr[ytr[:, 7]>0, :]
        x = np.r_[xtr1, xtr7]
        y = np.r_[np.zeros(xtr1.shape[0]), np.ones(xtr7.shape[0])]
        return x, y

    
class NewsModule(DataModule):
    def __init__(self, normalize=True, append_one=False):
        DataModule.__init__(self, normalize, append_one)
    
    def load(self):
        categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware']
        newsgroups_train = fetch_20newsgroups(
            subset='train', remove=('headers', 'footers', 'quotes'), categories=categories)
        newsgroups_test = fetch_20newsgroups(
            subset='test', remove=('headers', 'footers', 'quotes'), categories=categories)
        vectorizer = TfidfVectorizer(stop_words='english', min_df=0.001, max_df=0.20)
        vectors = vectorizer.fit_transform(newsgroups_train.data)
        vectors_test = vectorizer.transform(newsgroups_test.data)
        x1 = vectors
        y1 = newsgroups_train.target
        x2 = vectors_test
        y2 = newsgroups_test.target
        x = np.array(np.r_[x1.todense(), x2.todense()])
        y = np.r_[y1, y2]
        return x, y


class AdultModule(DataModule):
    def __init__(self, normalize=True, append_one=False, csv_path='./data'):
        DataModule.__init__(self, normalize, append_one)
        self.csv_path = csv_path
        
    def load(self):
        train = pd.read_csv('%s/adult-training.csv' % (self.csv_path,), names=columns)
        test = pd.read_csv('%s/adult-test.csv' % (self.csv_path,), names=columns, skiprows=1)
        df = pd.concat([train, test], ignore_index=True)

        # preprocess
        df.replace(' ?', np.nan, inplace=True)
        df['Income'] = df['Income'].apply(lambda x: 1 if x in (' >50K', ' >50K.') else 0)
        df['Workclass'].fillna(' 0', inplace=True)
        df['Workclass'].replace(' Without-pay', ' Never-worked', inplace=True)
        df['fnlgwt'] = df['fnlgwt'].apply(lambda x: np.log1p(x))
        df['Education'] = df['Education'].apply(primary)
        df['Marital Status'].replace(' Married-AF-spouse', ' Married-civ-spouse', inplace=True)
        df['Occupation'].fillna(' 0', inplace=True)
        df['Occupation'].replace(' Armed-Forces', ' 0', inplace=True)
        df['Native country'].fillna(' 0', inplace=True)
        df['Native country'] = df['Native country'].apply(native)

        # one-hot encoding
        categorical_features = df.select_dtypes(include=['object']).axes[1]
        for col in categorical_features:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col, prefix_sep=':')], axis=1)
            df.drop(col, axis=1, inplace=True)
        
        # data
        x = df.drop(['Income'], axis=1).values
        y = df['Income'].values
        return x, y
