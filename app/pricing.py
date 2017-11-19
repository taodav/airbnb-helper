
# coding: utf-8

# In[15]:

# HUGE shout out to https://www.kaggle.com/headsortails/pytanic for the guide. Extremely in depth feature decomposition
#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
from re import sub
from decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn import svm

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')


# In[16]:

data = pd.read_csv('./Economics/Real Estate/Airbnb Seattle Listings/listings.csv')


# In[17]:

data["neighbourhood_cleansed"].unique()
features = ['neighbourhood_cleansed', 'property_type','longitude','latitude', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'require_guest_profile_picture', 'calculated_host_listings_count', 'review_scores_rating']
df = data[features]
df = df.loc[df['review_scores_rating'].notnull()]


# In[18]:

field_index_mapping = {
    "neighbourhood_cleansed": "neighbourhood_indexed",
    "property_type": "property_indexed",
    "room_type": "room_indexed",
    "bed_type": "bed_indexed",
    "require_guest_profile_picture": "require_guest_profile_picture"
}
for field in ["neighbourhood_cleansed", "property_type", "room_type", "bed_type", "require_guest_profile_picture"]:
    val_types = df[field].unique()
    val_map = dict.fromkeys(val_types)
    for i, el in enumerate(val_types):
        val_map[el] = i
    df[field_index_mapping[field]] = df[field].map(val_map)


# In[19]:

for types in ["bathrooms", "bedrooms", "beds"]:
    df[types] = df[types].apply(lambda x: 1 if np.isnan(x) else x)
    df[types] = df[types].astype(int)


# In[20]:

df["price"] = df["price"].apply(lambda x: float(Decimal(sub(r'[^\d.]', '', x))))


# In[21]:

def parse_amenity(s):
    s = s[2:-2]
    s = s.replace('"', '')
    res = s.split(",")
    return len(res)
    print(len(res))
parse_amenity(df["amenities"][13])

df['amen_count'] = df["amenities"].map(parse_amenity)


# In[22]:

drop_features = ['neighbourhood_cleansed', 'property_type', 'room_type', 'bed_type', 'amenities', 'require_guest_profile_picture', 'calculated_host_listings_count']
train_input = df.drop(drop_features, axis=1)


# In[23]:

listings = train_input

bed_price = listings[['bedrooms', 'price']].groupby(['bedrooms'], as_index=False).mean()
bath_price = listings[['bathrooms', 'price']].groupby(['bathrooms'], as_index=False).mean()
amen_price = listings[['amen_count', 'price']].groupby(['amen_count'], as_index=False).mean()

bed_bath_price = listings[['bedrooms', 'bathrooms', 'price']].groupby(['bedrooms','bathrooms'], as_index=False).mean()

print(bed_price)

sns.set(style='white', context='notebook', palette='deep')
plt.figure()

plt.subplot(332)
sns.barplot('bedrooms', 'price', data=bed_price)

plt.subplot(333)
sns.barplot('bathrooms', 'price', data=bath_price)

plt.subplot(334)
sns.barplot('amen_count', 'price', data=amen_price)
plt.show()


# In[28]:

x = listings.drop(['price'], axis=1)
y = listings['price']
print(x.info())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print("Number of Training Samples: " + str(x_train.shape[0]))
print("Number of features: " + str(x_train.shape[1]))


# In[25]:

#Training the classifier
clf = GradientBoostingRegressor(learning_rate=0.25, max_depth=3, loss='ls', n_estimators=100)

clf.fit(x_train, y_train)
print("Accuracy on training set: {:.3f}".format(clf.score(x_train, y_train)))
print("Accuracy on training set: {:.3f}".format(clf.score(x_test, y_test)))


# In[26]:

pred = clf.predict(x_test[0:10])
print(pred)
print(y_test[0:10])


# In[27]:

with open('gbr.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:



