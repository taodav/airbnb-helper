# -*- coding: utf-8 -*-

from flask import Flask, request
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
  request.args.get('id')

@app.route('/')
def hello_world():
  return 'Hello, World!'

app.run(debug=True)
