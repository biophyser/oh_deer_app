from oh_deer import app
from flask import render_template
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import numpy as np
from mapbox import Geocoder, Directions
import pickle
from shapely.geometry import Point, LineString
import time
import geopandas
import overpass
from shapely.geometry import LineString, Point



year_file = open('oh_deer/static/yearly_trend_2010_2022.csv','r')
test = pd.read_csv(year_file, index_col=0)
test.index = pd.to_datetime(test.index)
loaded_model = pickle.load(open("oh_deer/static/deer_pred.pickle.dat", "rb"))
api_pass = overpass.API(timeout=600)

def deer_date(date):
    """
    Given a date give a probability of seeing deer.
    """

def gaussian(x, height, center, width, offset):
    return height*np.exp(-(x - center)**2/(2*width**2)) + offset

def three_gaussians(x, h1, c1, w1, h2, c2, w2, h3, c3, w3, offset):
    return (gaussian(x, h1, c1, w1, offset=0) +
        gaussian(x, h2, c2, w2, offset=0) +
        gaussian(x, h3, c3, w3, offset=0) + offset)

def deer_time(time):
    """
    Given a time of day give a probability of seeing deer.
    """

    time_query = time*60*60
    optim3 = np.array([3.32398040e-01, 1.46998346e+04, 5.49206304e+03, 7.06000543e-01,
       4.94612344e+04, 2.23484799e+03, 6.28245838e-01, 6.14473945e+04,
       7.46462554e+03, 1.34156912e-01])
    prob = three_gaussians(time_query, *optim3)
    return prob


def make_predictions(array):
    return loaded_model.predict(array[1:,:])


def split_line(line, max_line_units=5*(0.07381197609536122)):
    """max_line_units=0.07381197609536122
    is set for 5 mile increments"""
    if line.length <= max_line_units:
        return [line]

    half_length = line.length / 2
    coords = list(line.coords)
    for idx, point in enumerate(coords):
        proj_dist = line.project(Point(point))
        if proj_dist == half_length:
            return [LineString(coords[:idx + 1]), LineString(coords[idx:])]

        if proj_dist > half_length:
            mid_point = line.interpolate(half_length)
            head_line = LineString(coords[:idx] + [(mid_point.x, mid_point.y)])
            tail_line = LineString([(mid_point.x, mid_point.y)] + coords[idx:])
            return split_line(head_line, max_line_units) + split_line(tail_line, max_line_units)

def colorizer(value):
    # Colors red to blue
    colors = ['#d90026','#8c0073','#5900a6','#0d00f2']
    if value >= 0.5:
        color = 'red'
    # elif 0.75 >= value > 0.5:
    #     color = colors[1]
    # elif 0.5 >= value > 0.25:
    #     color = colors[2]
    else:
        color = 'blue'

    return color

def process_line_colors(predictions):
    segment_len = 1 / len(predictions)
    color_list = [colorizer(pred) for pred in predictions]
    segment_list = [n/(len(predictions)-1) for n in range(len(predictions))]
    #color_list = ['{}, "{}"'.format(n/(len(predictions)-1), colorizer(pred)) for n, pred in enumerate(predictions)]
    #color_string = ', '.join(color_list)
    colors_segments = [color_list, segment_list]
    target = '0, "blue", 0.1, "blue", 0.3, "blue", 0.5, "red", 0.7, "red", 1, "blue"'
    return colors_segments

def overpass_query(line_list):
    # There is an index column here
    feature_df_list = [pd.read_csv('oh_deer/static/dummy_features.csv', index_col=0)]
    cols = feature_df_list[0].columns
    print("LINES:{}".format(len(line_list)))
    for n, line in enumerate(line_list):
        print("LINE:{}".format(n))
        poly = line.buffer(.0025, cap_style=3)
        simplified = poly.simplify(0.001, preserve_topology=False)
        
        # Building overpass queries
        road_query = 'nwr(around:1, {c[0]}, {c[1]});(._;>;);out;'.format(
            c=line.interpolate(0.5, normalized = True).coords[:][0])
        area_query = 'nwr(poly:"{}");out;'.format(
            ' '.join('{c[0]} {c[1]}'.format(c=c) for c in simplified.exterior.coords[:]))
        
        # Getting overpass road responses
        road_response = api_pass.get(road_query)
        road_geodf = geopandas.GeoDataFrame.from_features(road_response)
        culled_road = road_geodf.dropna(axis='index', how='all', subset=road_geodf.columns[1:])
        # Overwriting feature_dict each time
        feature_dict = make_feature_dict_road(culled_road, ['highway', 'surface'], prefix='road_')
        time.sleep(4)
        
        # Getting overpass area responses
        response = api_pass.get(area_query)
        geodf = geopandas.GeoDataFrame.from_features(response)
        culled_geodf = geodf.dropna(axis='index', how='all', subset=geodf.columns[1:])
        feature_dict.update(make_feature_dict_area(culled_geodf, cols, prefix=None))
        feature_df_list.append(pd.io.json.json_normalize(feature_dict))
        time.sleep(2)
    feature_df = pd.concat(feature_df_list, sort=True)
    feature_df.reset_index(inplace=True)
    return feature_df

@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Jeremy' },
       )

@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html", lat=-72.883145, lon=43.858205)

@app.route('/output', methods=['GET', 'POST'])
def output():
    # get origin and destination geolocations
    key = 'pk.eyJ1IjoiZGF0YXNsZXV0aCIsImEiOiJjazB0em1tbGUwaXdnM21yenJjdTJybm52In0.qm4lOhweUJZuaxgEl6lEwA'
    geocoder = Geocoder()
    geocoder.session.params['access_token'] = key
    directions = Directions()
    directions.session.params['access_token'] = key

    
    startname = request.args.get('origin')
    endname = request.args.get('destination')
    if startname == '' or endname == '':
        startname = 'montpelier, vt'
        endname = 'salisbury, vt'
    
    startresponse = geocoder.forward(startname)
    endresponse = geocoder.forward(endname)
    origin = startresponse.geojson()['features'][0]
    destination = endresponse.geojson()['features'][0]
    response = directions.directions([origin, destination], 'mapbox/driving') 

    coords = response.geojson()['features'][0]['geometry']['coordinates']
    shapely_line = LineString(coords)
    line = '['+','.join(["[{},{}]".format(lat, lon) for lat, lon in coords])+']'

    # Splitting shapely_line
    line_list = split_line(shapely_line)
    # Get features from overpass
    feature_df = overpass_query(line_list)
    feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
    feature_df.to_csv('debugging.csv')
    # Make prediction
    #predictions = make_predictions(feature_df.values)
    predictions = [.2, .7, .8, .2]
    print(predictions)
    # Make pretty colors
    #colorstring = process_line_colors(predictions)
    colors_segments = process_line_colors(predictions)
    print(colors_segments)
    #print('COLORSTRING\n{}'.format(colors_segments))

    #pull 'date' from input field and store it
    drive_date = request.args.get('date')
    drive_date = pd.to_datetime(drive_date)
    hour, minute = request.args.get('time').split(":")
    minute = float(minute) / 60
    hour = float(hour) + minute
    #drive_time = int(request.args.get('time'))/100

    
    idx = test.index.get_loc(drive_date, method='nearest')
    date_prob = test.iloc[idx].values[0]
    time_prob = deer_time(hour)
    time_multiplier = (date_prob * time_prob * 1 )
    if time_multiplier > 0.5:
        color = "red"
    else:
        color = "blue"

    #print(color)

    return render_template("output.html", line=line, lat=-72.883145, lon=43.858205, colors_segments=colors_segments)


def make_feature_dict_road(df, features, prefix=None):
    """
    
    Parameters
    ----------
    df : geopandas.geodataframe.GeoDataFrame
        Dataframe containing features pulled from overpass API
        
    features : list
        List of column names associated with high-level feature names
        employed by overpass/OSM. Reference here:
        https://wiki.openstreetmap.org/wiki/Map_Features
        
    prefix : str
        Additional identifier placed before feature name.
        
    Returns
    -------
    dict
        Nested dictionary of features with value counts as eventual values.
        
    """
    
    if prefix!=None:
        pass
    else:
        prefix=''
        
    feature_dict = {}
    for feature in features:
        if feature in df.columns:
            series = df[feature].value_counts()
            feature_dict[prefix+feature] = { k:v for (k,v) in zip(series.index, series.values)}
        #else:
            #feature_dict[prefix+feature] = None
    return feature_dict

def make_feature_dict_area(df, features, prefix=None):
    """
    
    Parameters
    ----------
    df : geopandas.geodataframe.GeoDataFrame
        Dataframe containing features pulled from overpass API
        
    features : list
        List of column names associated with high-level feature names
        employed by overpass/OSM. Reference here:
        https://wiki.openstreetmap.org/wiki/Map_Features
        
    prefix : str
        Additional identifier placed before feature name.
        
    Returns
    -------
    dict
        Nested dictionary of features with value counts as eventual values.
        
    """
    
    if prefix!=None:
        pass
    else:
        prefix=''
    feature_dict = {}
    for col in df.columns:
        col_prefix = prefix+col
        if col_prefix in features:
            series = df[col].value_counts()
            feature_dict[col_prefix] = { k:v for (k,v) in zip(series.index, series.values)}
        else:
            feature_dict[col_prefix] = None
        
    return feature_dict