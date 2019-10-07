from oh_deer import app
from flask import render_template
import pandas as pd
from flask import request
import numpy as np
from mapbox import Geocoder, Directions
import pickle
from shapely.geometry import Point, LineString
import time
import geopandas
import overpass
from shapely.geometry import LineString, Point
from oh_deer import config



year_file = open('oh_deer/static/yearly_trend_2010_2022.csv','r')
test = pd.read_csv(year_file, index_col=0)
test.index = pd.to_datetime(test.index)
#loaded_model = pickle.load(open("oh_deer/static/deer_pred.pickle.dat", "rb"))
loaded_model = pickle.load(open("oh_deer/static/pa_deer_pred.pickle.dat", "rb"))
api_pass = overpass.API(timeout=600)

dummy_df = pd.read_csv('oh_deer/static/dummy_features.csv', index_col=0)

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
    # http://www.perbang.dk/rgbgradient/
    colors = ['#6DF7F8','#97A5F9','#C254FA','#ED03FC']
    if value > 0.65:
        color = colors[3]
    elif 0.65 >= value > 0.52:
        color = colors[2]
    elif 0.52 >= value > 0.37:
        color = colors[1]
    else:
        color = colors[0]

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
    feature_df_list = list()
    print("LINES:{}".format(len(line_list)))
    for n, line in enumerate(line_list[:]):
        print("LINE:{}".format(n))
        poly = line.buffer(.001, cap_style=3)
        simplified = poly.simplify(0.001, preserve_topology=False)

        # Building overpass queries
        midpoint = line.interpolate(0.5, normalized = True).coords[:][0]
        
        buffer = 5e-3
        road_query = 'nwr[highway]({}, {}, {}, {});(._;>;);out;'.format(midpoint[1]-buffer, midpoint[0]-buffer, midpoint[1]+buffer, midpoint[0]+buffer)
        area_query = 'nwr(poly:"{}");out;'.format(
            ' '.join('{c[1]} {c[0]}'.format(c=c) for c in simplified.exterior.coords[:]))

        # Getting overpass road responses
        road_response = api_pass.get(road_query)
        road_geodf = geopandas.GeoDataFrame.from_features(road_response)
        culled_road = road_geodf.dropna(axis='index', how='all', subset=road_geodf.columns[1:])
        print(culled_road.shape)
        # Overwriting feature_dict each time
        feature_dict = make_feature_dict(culled_road, ['highway', 'surface'], prefix='road_')
        #feature_dict = make_feature_dict_road(culled_road, ['highway', 'surface'], prefix='road_')
        time.sleep(5)

        # Getting overpass area responses
        response = api_pass.get(area_query)
        geodf = geopandas.GeoDataFrame.from_features(response)
        culled_geodf = geodf.dropna(axis='index', how='all', subset=geodf.columns[1:])
        print(culled_geodf.shape)
        feature_dict.update(make_feature_dict(culled_geodf, cols, prefix='area_'))
        #feature_dict.update(make_feature_dict_area(culled_geodf, cols, prefix=None))
        feature_df_list.append(pd.io.json.json_normalize(feature_dict))
        time.sleep(5)
    feature_df = pd.concat(feature_df_list, ignore_index=True, sort=False)
    print(feature_df.shape)
    feature_df = pd.concat([dummy_df, feature_df], sort=False)[dummy_df.columns].iloc[1:,:]
    return feature_df


@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Jeremy' },
       )

@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html", lon=-95.281123, lat=38.762650)

@app.route('/output', methods=['GET', 'POST'])
def output():
    # get origin and destination geolocations
    key = config.mapbox['key']
    geocoder = Geocoder()
    geocoder.session.params['access_token'] = key
    directions = Directions()
    directions.session.params['access_token'] = key

    
    startname = request.args.get('origin')
    endname = request.args.get('destination')
    if startname == '' or endname == '':
        startname = 'Punxsutawney, PA'
        endname = 'State College, PA'
    
    startresponse = geocoder.forward(startname)
    endresponse = geocoder.forward(endname)
    origin = startresponse.geojson()['features'][0]
    destination = endresponse.geojson()['features'][0]
    response = directions.directions([origin, destination], 'mapbox/driving') 

    coords = response.geojson()['features'][0]['geometry']['coordinates']
    shapely_line = LineString(coords)
    midpoint = shapely_line.interpolate(0.5, normalized=True).coords[:][0]
    line = '['+','.join(["[{},{}]".format(lat, lon) for lat, lon in coords])+']'

    # Splitting shapely_line
    line_list = split_line(shapely_line)
    # Get features from overpass
    feature_df = overpass_query(line_list)
    #feature_df = feature_df.reindex(sorted(feature_df.columns), axis=1)
    feature_df.to_csv('debugging.csv')
    # Make prediction
    predictions = loaded_model.predict(feature_df)
    #predictions = [.2, .7, .8, .2]
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

    return render_template("output.html", line=line, lat=midpoint[0], lon=midpoint[1], colors_segments=colors_segments)


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


cols = [
    'aeroway',
    'amenity',
    'barrier',
    'boundary',
    'building',
    'healthcare',
    'highway',
    'landuse',
    'leisure',
    'man_made',
    'natural',
    'parking',
    'power',
    'railway',
    'route',
    'service',
    'surface',
    'tourism',
    'waterway'
]

def make_feature_dict(df, features, prefix=None):
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
        else:
            feature_dict[prefix+feature] = None
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