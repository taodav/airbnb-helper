
# coding: utf-8

# In[242]:

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
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.graph_objs import *
import pickle

plotly.tools.set_credentials_file(username='taodav', api_key='0Wy5MNzCH5IfB7AHg2zM')


from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

sns.set(style='white', context='notebook', palette='deep')


# In[228]:

data = pd.read_csv('./Economics/Real Estate/Airbnb Seattle Listings/listings.csv')
print(data.info())


# In[229]:

features = ['listing_url', 'neighbourhood_cleansed', 'weekly_price', 'monthly_price', 'property_type','longitude','latitude', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'price', 'require_guest_profile_picture', 'calculated_host_listings_count', 'review_scores_rating']
df = data[features]
df = df.loc[df['review_scores_rating'].notnull()]
df.info()

#listing_url                       3171 non-null object
#neighbourhood_cleansed            3171 non-null object
#weekly_price                      1792 non-null object
#monthly_price                     1356 non-null object
#property_type                     3171 non-null object
#longitude                         3171 non-null float64
#latitude                          3171 non-null float64
#room_type                         3171 non-null object
#accommodates                      3171 non-null int64
#bathrooms                         3171 non-null int32
bedrooms                          3171 non-null int32
beds                              3171 non-null int32
bed_type                          3171 non-null object
amenities                         3171 non-null object
price                             3171 non-null float64
require_guest_profile_picture     3171 non-null int64
calculated_host_listings_count    3171 non-null int64
review_scores_rating              3171 non-null float64

# we want to predict review_scores_rating


# In[230]:

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
df.info()


# In[231]:

for types in ["bathrooms", "bedrooms", "beds"]:
    df[types] = df[types].apply(lambda x: 1 if np.isnan(x) else x)
    df[types] = df[types].astype(int)
df.info()


# In[232]:

# processing prices
df["price"] = df["price"].apply(lambda x: float(Decimal(sub(r'[^\d.]', '', x))))
df["has_monthly_price"] = df["monthly_price"].isnull() == False
df["has_weekly_price"] = df["weekly_price"].isnull() == False


# In[233]:

# for i, am in enumerate(df["amenities"]):

def parse_amenity(s):
    s = s[2:-2]
    s = s.replace('"', '')
    res = s.split(",")
    return res
parse_amenity(df["amenities"][13])
df["amenities"] = df["amenities"].apply(parse_amenity)
df["amenities"][1]


# In[246]:

df["event_suitable"] = df["amenities"].apply(lambda x: 1 if 'Suitable for Events' in x else 0)
df["event_suitable"].unique()
df["event_suitable"].value_counts()
df.info()


# In[235]:

drop_features = ['neighbourhood_cleansed', 'listing_url', 'weekly_price', 'monthly_price', 'property_type', 'room_type', 'bed_type', 'amenities', 'require_guest_profile_picture', 'calculated_host_listings_count']
train_input = df.drop(drop_features, axis=1)
print(train_input.info())


# In[241]:

scaler = MinMaxScaler()
svd = TruncatedSVD(n_components=3)

scaler.fit(train_input)

train_scaled = scaler.transform(train_input)

train_pca.shape

encoder = OneHotEncoder()

encoder.fit(train_scaled)

train_encoded = encoder.transform(train_scaled)

train_svd = svd.fit_transform(train_encoded)


# In[244]:

# est = KMeans(n_clusters=5)
# est.fit(df_std)
# labels = est.labels_
agg = AgglomerativeClustering(n_clusters=5)
labels = agg.fit_predict(train_svd)
clf = LinearSVC()
clf.fit(train_svd, labels)

with open('agg.pickle', 'wb') as handle:
    pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
# now clf.predict(input) will give you the correct label
labels.shape


# In[215]:

fun = np.vectorize((lambda x: x * 2))
trace1 = go.Scatter3d(
#     x=df[features[0]],
#     y=df[features[1]],
#     z=df[features[2]],
    x=train_svd[:, 0],
    y=train_svd[:, 1],
    z=train_svd[:, 2],
    mode='markers',
    marker=dict(
        size=2,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        color=fun(labels).astype(np.float),
        opacity=0.8
    )
)
layout = go.Layout(
    scene=Scene(
    xaxis=dict(
        title='X'
    ),
    yaxis=dict(
        title='Y'
    ),
    zaxis=dict(
        title='Z'
    ),),
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=[trace1], layout=layout)
py.iplot(fig, filename='simple-3d-scatter')


# In[223]:

df["category"] = labels
df_listing_cat = df[['listing_url', 'category']]
df_listing_cat.loc[df_listing_cat["category"]==3]


# In[ ]:

category_mappings = {
    2: "modern",
    1: "luxury",
    4: "tourist",
    0: "homey",
    3: "quiet"
}


# In[245]:

with open('agg.pickle', 'rb') as handle:
    agg1 = pickle.load(handle)


# In[ ]:

agg1.predict(df.iloc)

