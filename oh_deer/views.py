from oh_deer import app
from flask import render_template
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import numpy as np
from mapbox import Geocoder, Directions

year_file = open('oh_deer/static/yearly_trend_2010_2022.csv','r')
test = pd.read_csv(year_file, index_col=0)
test.index = pd.to_datetime(test.index)

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


@app.route('/index')
def index():
    return render_template("index.html",
       title = 'Home', user = { 'nickname': 'Jeremy' },
       )

@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html", lat=-72.883145, lon=43.858205)

@app.route('/output')
def deer_output():
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
    print(response)   

    #pull 'date' from input field and store it
    drive_date = request.args.get('date')
    drive_date = pd.to_datetime(drive_date)
    print(drive_date)
    drive_time = int(request.args.get('time'))/100

    
    idx = test.index.get_loc(drive_date, method='nearest')
    date_prob = test.iloc[idx].values[0]
    time_prob = deer_time(drive_time)
    the_result = (date_prob * time_prob * 9 ) + 1
    #the_result = type(time_prob)

    return render_template("output.html", the_result=the_result)