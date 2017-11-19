# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
import pickle
import json
import requests

app = Flask(__name__)

# longitude                3171 non - null float64
# latitude                 3171 non - null float64
# accommodates             3171 non - null int64
# bathrooms                3171 non - null int32
# bedrooms                 3171 non - null int32
# beds                     3171 non - null int32
# review_scores_rating     3171 non - null float64
# neighbourhood_indexed    3171 non - null int64
# property_indexed         3171 non - null int64
# room_indexed             3171 non - null int64
# bed_indexed              3171 non - null int64
# amen_count               3171 non - null int64
#  - for gbr

# longitude                3171 non - null float64
# latitude                 3171 non - null float64
# accommodates             3171 non - null int64
# bathrooms                3171 non - null int32
# bedrooms                 3171 non - null int32
# beds                     3171 non - null int32
# price                    3171 non - null float64
# review_scores_rating     3171 non - null float64
# neighbourhood_indexed    3171 non - null int64
# property_indexed         3171 non - null int64
# room_indexed             3171 non - null int64
# bed_indexed              3171 non - null int64
# has_monthly_price        3171 non - null bool
# has_weekly_price         3171 non - null bool
# event_suitable           3171 non - null int64
#  - for agg

@app.route('/pred')
def pred():
  id = request.args.get('id')
  payload = {
    "client_id": "3092nxybyb0otqw18e8nh5nty",
    "locale": "en-US",
    "currency": "CAD",
    "_format": "v1_legacy_for_p3",
    "_source": "mobile_p3"
  }
  r = requests.get('https://api.airbnb.com/v2/listings/' + id, params=payload)
  res = r.json()["listing"]

  review_scores_rating = (res["review_rating_accuracy"] + \
      res["review_rating_checkin"] + res["review_rating_cleanliness"] + \
      res["review_rating_communication"] + res["review_rating_location"] + \
      res["review_rating_value"]) / 6

  with open("keys.json", 'r') as f:
          keys = json.load(f)
  gbr_input = {
    "longitude": res["lng"],
    "latitude": res["lat"],
    "accommodates": res["person_capacity"],
    "bathrooms": res["bathrooms"],
    "bedrooms": res["bedrooms"],
    "beds": res["beds"],
    "review_scores_rating": review_scores_rating,
    "neighbourhood_indexed": keys["neighbourhood"][res["neighborhood"]],
    "property_indexed": keys["property_type"][res["property_type"]],
    "room_indexed": keys["room_type"][res["room_type"]],
    "bed_indexed": keys["bed_type"][res["bed_type"]],
    "amen_count": len(res["amenities"])
  }
  event_suitable = True if "Suitable for events" in res["amenities"] else False
  agg_input = {
    "longitude": gbr_input["longitude"],
    "latitude": gbr_input["latitude"],
    "accommodates": gbr_input["accommodates"],
    "bathrooms": gbr_input["bathrooms"],
    "bedrooms": gbr_input["bedrooms"],
    "beds": gbr_input["beds"],
    "price": res["price"],
    "review_scores_rating": review_scores_rating,
    "neighbourhood_indexed": gbr_input["neighbourhood_indexed"],
    "property_indexed": gbr_input["property_indexed"],
    "room_indexed": gbr_input["room_indexed"],
    "bed_indexed": gbr_input["bed_indexed"],
    "has_monthly_price": bool(res['monthly_price_native']),
    "has_weekly_price": bool(res["weekly_price_native"]),
    "event_suitable": event_suitable
  }
  category_mappings = {
      2: "modern",
      1: "luxury",
      4: "tourist",
      0: "homey",
      3: "quiet"
  }

  df_gbr = pd.DataFrame(gbr_input, index=[0])
  df_agg = pd.DataFrame(agg_input, index=[0])

  with open('agg.pickle', 'rb') as handle:
    agg = pickle.load(handle)

  with open('gbr.pickle', 'rb') as handle:
    gbr = pickle.load(handle)

  with open('encoder.pickle', 'rb') as handle:
    encoder = pickle.load(handle)

  with open('scaler.pickle', 'rb') as handle:
    scaler = pickle.load(handle)

  with open('svd.pickle', 'rb') as handle:
    svd = pickle.load(handle)
  
  train_scaled = scaler.transform(df_agg)
  print(train_scaled)
  train_encoded = encoder.transform(train_scaled)
  train_svd = svd.fit_transform(train_encoded)
  

  data = {
    "price": abs(gbr.predict(df_gbr)[0]),
    "type": category_mappings[agg.predict(train_svd)[0]]
  }
  return jsonify(data)

@app.route('/')
def hello_world():
  return 'Hello, World!'

app.run(debug=True)
