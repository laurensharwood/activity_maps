#!/usr/bin/ python

"""
in linux/mac terminal: 
export GARMIN_EMAIL=your garmin username/email 
export GARMIN_PWD=your garmin pwd
export POSTPWD=user/postgresql password
"""
import os, sys
import shutil
import numpy as np
import pandas as pd
import json
import logging
from getpass import getpass
from enum import Enum, auto
from typing import Any, Dict, List, Optional
import requests
import garth
from garth.exc import GarthHTTPError

import datetime
from datetime import datetime, timedelta, date, timezone
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import geopandas as gpd
from shapely.geometry import Point, LineString
from datetime import datetime, timedelta, date, timezone
## under GEO, for transform_point_coords
from pyproj import Proj, transform
## parsing TCX/GPX functions
from tcxreader.tcxreader import TCXReader
import psycopg2
## route heatmap
import folium
from folium import plugins
## 3d plot
import plotly.express as px
## under RUNNING ACTIVITIES, cal_heatmap function
import matplotlib.pyplot as plt
import calplot ## https://github.com/tomkwok/calplot.git (pip install calplot)
import time


class Garmin:
    """
    from https://github.com/cyberjunky/python-garminconnect/blob/master/garminconnect/__init__.py
    Class for fetching data from Garmin Connect.
    """

    def __init__(self, email=None, password=None, is_cn=False, prompt_mfa=None):
        """Create a new class instance."""
        self.username = email
        self.password = password
        self.is_cn = is_cn
        self.prompt_mfa = prompt_mfa
        self.garmin_connect_activities = ("/activitylist-service/activities/search/activities")
        self.garmin_connect_activity = "/activity-service/activity"
        self.garmin_connect_activity_types = ("/activity-service/activity/activityTypes" )
        self.garmin_connect_fit_download = "/download-service/files/activity"
        self.garmin_connect_tcx_download = ("/download-service/export/tcx/activity")
        self.garmin_connect_gpx_download = ("/download-service/export/gpx/activity")
        self.garmin_connect_kml_download = ("/download-service/export/kml/activity")
        self.garmin_connect_csv_download = ("/download-service/export/csv/activity")
        self.garmin_workouts = "/workout-service"
        self.garth = garth.Client(domain="garmin.cn" if is_cn else "garmin.com")
        self.display_name = None
        self.full_name = None
        self.unit_system = None

    def connectapi(self, path, **kwargs):
        return self.garth.connectapi(path, **kwargs)

    def download(self, path, **kwargs):
        return self.garth.download(path, **kwargs)

    def login(self, /, tokenstore: Optional[str] = None):
        """Log in using Garth."""
        tokenstore = tokenstore or os.getenv("GARMINTOKENS")
        if tokenstore:
            if len(tokenstore) > 512:
                self.garth.loads(tokenstore)
            else:
                self.garth.load(tokenstore)
        else:
            self.garth.login(self.username, self.password, prompt_mfa=self.prompt_mfa)
        self.display_name = self.garth.profile["displayName"]
        self.full_name = self.garth.profile["fullName"]
        return True

    def get_activities_by_date(self, startdate, enddate, activitytype=None):
        """
        Fetch available activities between specific dates
        :param startdate: String in the format YYYY-MM-DD
        :param enddate: String in the format YYYY-MM-DD
        :param activitytype: (Optional) Type of activity you are searching
                             Possible values are [cycling, biking]
        :return: list of JSON activities
        """

        activities = []
        start = 0
        limit = 20
        # mimicking the behavior of the web interface that fetches
        # 20 activities at a time
        # and automatically loads more on scroll
        url = self.garmin_connect_activities
        params = {
            "startDate": str(startdate),
            "endDate": str(enddate),
            "start": str(start),
            "limit": str(limit),
        }
        if activitytype:
            params["activityType"] = str(activitytype)
        logger.debug( f"Requesting activities by date from {startdate} to {enddate}")
        while True:
            params["start"] = str(start)
            logger.debug(f"Requesting activities {start} to {start+limit}")
            act = self.connectapi(url, params=params)
            if act:
                activities.extend(act)
                start = start + limit
            else:
                break
        return activities

    class ActivityDownloadFormat(Enum):
        """Activity variables."""
        ORIGINAL = auto()
        TCX = auto()
        GPX = auto()
        KML = auto()
        CSV = auto()

    def download_activity(self, activity_id, dl_fmt=ActivityDownloadFormat.TCX):
        """
        Downloads activity in requested format and returns the raw bytes. For
        "Original" will return the zip file content, up to user to extract it.
        "CSV" will return a csv of the splits.
        """
        activity_id = str(activity_id)
        urls = {
            Garmin.ActivityDownloadFormat.ORIGINAL: f"{self.garmin_connect_fit_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.TCX: f"{self.garmin_connect_tcx_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.GPX: f"{self.garmin_connect_gpx_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.KML: f"{self.garmin_connect_kml_download}/{activity_id}",  # noqa
            Garmin.ActivityDownloadFormat.CSV: f"{self.garmin_connect_csv_download}/{activity_id}",  # noqa
        }
        if dl_fmt not in urls:
            raise ValueError(f"Unexpected value {dl_fmt} for dl_fmt")
        url = urls[dl_fmt]
        logger.debug("Downloading activities from %s", url)
        return self.download(url)

class GarminConnectConnectionError(Exception):
    """Raised when communication ended in error."""

class GarminConnectTooManyRequestsError(Exception):
    """Raised when rate limit is exceeded."""

class GarminConnectAuthenticationError(Exception):
    """Raised when authentication is failed."""

class GarminConnectInvalidFileFormatError(Exception):
    """Raised when an invalid file format is passed to upload."""

def display_json(api_call, output):
    """Format API output for better readability."""
    dashed = "-" * 20
    header = f"{dashed} {api_call} {dashed}"
    footer = "-" * len(header)
    print(header)
    if isinstance(output, (int, str, dict, list)):
        print(json.dumps(output, indent=4))
    else:
        print(output)
    print(footer)

def display_text(output):
    """Format API output for better readability."""
    dashed = "-" * 60
    header = f"{dashed}"
    footer = "-" * len(header)

def get_credentials():
    """Get user credentials."""
    email = input("Login e-mail: ")
    password = getpass("Enter password: ")
    return email, password


def init_api(email, password, tokenstore):
    '''
    Initialize Garmin API with your credentials.
    '''
    try:
        # if not email or not password:
        #     email, password = get_credentials()
       ## print(f"Trying to login to Garmin Connect using token data from ...\n'{tokenstore}'" )
        garmin = Garmin()
        garmin.login(tokenstore)
        garth.save('~/.garth')

    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again
       ## print(  Login tokens not present, login with your Garmin Connect credentials to generate them.\n'  f"They will be stored in "{tokenstore}" for future use.\n")
        email, password = get_credentials()
        garmin = Garmin(email, password)
        garmin.login()
        garth.save('~/.garth')
        try:
            garmin = Garmin(email, password)
            garmin.login()
            garth.save('~/.garth')

            # Save tokens for next login
            garmin.garth.dump(tokenstore)

        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, requests.exceptions.HTTPError) as err:
            logger.error(err)
            return None

    return garmin

######################################################################

def switch(api, option, out_dir, startdate, today):
    '''
    Run selected API call.
    '''
    if api:
        activitytype=''
        activities = api.get_activities_by_date(startdate.isoformat(), today.isoformat(), activitytype )
        for activity in activities:
            activity_id = activity['activityId']
            activity_name = activity['activityName']
            activity_start = activity['startTimeLocal'].replace(' ', '', -1).replace(':', '', -1).replace('-', '', -1)            
            if option=='.gpx':
                gpx_data = api.download_activity( activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.gpx"), 'wb') as fb:
                    fb.write(gpx_data)
            elif option=='.tcx':
                tcx_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.TCX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.tcx"), 'wb') as fb:
                    fb.write(tcx_data)
            elif option=='.zip':
                zip_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
                with open(os.path.join(out_dir, f"{str(activity_start)}.zip"), 'wb') as fb:
                    fb.write(zip_data)
            elif option== '.csv':
                csv_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.CSV)
                with open(os.path.join(out_dir, f"{str(activity_start)}.csv"), 'wb') as fb:
                    fb.write(csv_data)

def get_garmin(num_days, project_dir, file_types):
    '''
    navigate into project directory (location of script) & run from there
    file_types: all options = ['.tcx', '.gpx', '.csv', '.zip']
    saves activity files into a new folder named today's date (YYYYMMDD) located in the script/project directory
    '''
    ## keep python script in the project directory with archive & out folders
    ## Load environment variables with Garmin credentials

    api = None

    today = date.today()
    startdate = today - timedelta(days=int(num_days))
    YYYYMMDD = today.strftime('%Y%m%d')
    out_dir = os.path.join(project_dir, YYYYMMDD)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    garth.save('~/.garth')

    ## 1) download from garmin
    for option in file_types:
        menu_options = {
            option: f"Download activities data by date from '{startdate.isoformat()}' to '{today.isoformat()}'",
            "q": "Exit"}
        if not api:
            try:
                email = os.getenv('GARMIN_EMAIL')
                password = os.getenv('GARMIN_PWD')
                tokenstore = os.getenv('GARMINTOKENS') or '~/.garminconnect'
            except:
                email, password = get_credentials()
            tokenstore = '~/.garminconnect'
            api = init_api(email, password, tokenstore)
        if api:
            switch(api, option, out_dir, startdate, today)
        else:
            tokenstore = os.getenv('GARMINTOKENS') or '~/.garminconnect'
            api = init_api(email, password, tokenstore)
            switch(api, option, out_dir, startdate, today)
    if len(os.listdir(out_dir)) == 0:
        os.rmdir(out_dir)

    return out_dir

################################
## TCX / GPX ACTIVITY PARSING
################################
def tcx_to_df(tcx_files, subtr_hrs=6):
    '''
    parse all TCX files in data directory -- make directory name the date of last activity -- writes csv with that date
    function is quick so okay to reparse TCX files -- for next run, add new files to folder
    '''
    run_df = pd.DataFrame(columns=['filename', 'start',  'distance', 'duration', 'ascent', 'hr_max', 'hr_avg', 'avg_speed'])
    bike_df = pd.DataFrame(columns=['filename', 'start',  'distance', 'duration', 'ascent', 'hr_max', 'hr_avg', 'avg_speed'])
    for tf in tcx_files:
        file = open(tf, 'r')
        tcx_reader = TCXReader()
        exercise = tcx_reader.read(tf)
        if (exercise.activity_type == 'Running' and exercise.duration != None):
            run_df.loc[len(run_df.index)] = [os.path.basename(file.name), exercise.start_time, exercise.distance, exercise.duration, exercise.ascent, exercise.hr_max, exercise.hr_avg, exercise.avg_speed]
        elif (exercise.activity_type == 'Biking' and exercise.duration != None):
            bike_df.loc[len(bike_df.index)] = [os.path.basename(file.name), exercise.start_time, exercise.distance, exercise.duration, exercise.ascent, exercise.hr_max, exercise.hr_avg, exercise.avg_speed]
    run_df['start'] = run_df['start'].astype('datetime64[ns]') - timedelta(hours=6)
    bike_df['start'] = bike_df['start'].astype('datetime64[ns]') - timedelta(hours=6)
    return (run_df, bike_df)

def gpx_to_df(files):
    '''for google colab workflow -- faster than parse_gpx function, but only ele extracted (no speed), and doesn't split files'''
    df = pd.DataFrame(columns=['date', 'filename', 'lat', 'lon', 'ele', 'speed'])
    for fi in files:
        gpx = gpd.read_file(fi, layer='track_points')
        for k, v in gpx.iterrows():
          df.loc[len(df.index)] = [v.time, os.path.basename(fi), v.geometry.y, v.geometry.x, gpx.ele, 0]
    df['date'] = [str(i).split('+')[0] for i in df['date']]
    return df.sort_values('date')

def parse_gpx(files):
    df = pd.DataFrame(columns=['date', 'filename', 'lat', 'lon', 'ele', 'speed'])
    for fi in files:
        gpx_file = open(fi, 'r')
        gpx = gpxpy.parse(gpx_file, version='1.1')
        for track in gpx.tracks:
            for seg in track.segments:
                for point_no, pt in enumerate(seg.points):
                    if pt.speed != None:
                        speed = pt
                    elif point_no > 0:
                        speed = pt.speed_between(seg.points[point_no - 1])
                    else:
                        speed = 0
                    df.loc[len(df.index)] = [pt.time, os.path.basename(fi), pt.latitude, pt.longitude, pt.elevation, speed]
    df['date'] = [str(i).split('+')[0] for i in df['date']]
    return df.sort_values('date')

def split_gpx_at(fi, split_min):
    gpx_file = open(fi, 'r')
    gpx = gpxpy.parse(gpx_file, version='1.1')
    trackpoints = []
    trackpoints2 = []
    for track in gpx.tracks:
        for seg in track.segments:
            for point_no, pt in enumerate(seg.points):
                first_part = True
                if point_no == 0:
                    trackpoints.append([pt.time, os.path.basename(fi), pt.latitude, pt.longitude, pt.elevation, 0]) ## speed = 0
                elif point_no > 0:
                    speed = pt.speed_between(seg.points[point_no - 1])
                    secs_btwn = pt.time - seg.points[point_no - 1].time
                    minutes = secs_btwn.total_seconds() / 60
                    if minutes < split_min:
                        trackpoints.append([pt.time, os.path.basename(fi), pt.latitude, pt.longitude, pt.elevation, speed])
                    elif (minutes > split_min or first_part == False):
                        trackpoints2.append([pt.time, os.path.basename(fi.replace(".gpx", "_2.gpx")), pt.latitude, pt.longitude, pt.elevation, speed])
                        first_part = False
                    else:
                        print('CHECK')
    return (trackpoints, trackpoints2)

def gpx_to_postgres(gpx_files, table_name, db, usr='postgres', pwd='', host='localhost', port=5432):
    '''
    gpx_files  = files to parse and insert into postgres db
    db = database
    table_name = table in db that gpx waypoints will be added to (should already exist in db)
        CREATE TABLE table_name (filename CHAR(18) REFERENCES activity_names (activity_gpx), date TIMESTAMP PRIMARY KEY, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);
    returns list of files that were parsed
    '''
    try:
        with psycopg2.connect(database = db, user = usr, password = pwd, host = host, port = port) as conn:
            with conn.cursor() as cur:
                for fi in gpx_files:
                    gpx_file = open(fi, 'r')
                    gpx = gpxpy.parse(gpx_file, version='1.1')
                    for track in gpx.tracks:
                        for seg in track.segments:
                            for point_no, pt in enumerate(seg.points):
                                ## add _2 to filename if consecutive trackpoints are more than X minutes apart 
                                run_parts = split_gpx_at(fi = fi, split_min = 45)
                                for run_part in [i for i in run_parts if len(i) > 0]:
                                    for trackpt in run_part:
                                        trackpt[0] = trackpt[0].strftime("%Y-%m-%d %H:%M:%S")
                                        cur.execute('INSERT INTO '+table_name+' (date, filename, lat, lon, ele, speed) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                                                  [str(i) for i in trackpt])
                conn.commit()
            print('Records successfully inserted into '+table_name+' table within '+db+' db')
    except (Exception, psycopg2.Error) as error:
        print('Error while inserting GPX data into ', table_name, ' PostgreSQL table: ', error)
        conn.close()
        print('PostgreSQL connection is closed')        
    finally:
        if conn:
            conn.close()
            print('PostgreSQL connection is closed')

def tcx_to_postgres(tcx_files, db, usr='postgres', pwd='', host='localhost', port=5432, subtr_hrs=6):
    '''
    data_dir  = input directory to parse all TCX files in
    1) adds basic activity stats to 'run_stats' or 'bike_stats' table in 'tcx_activities' postgres database
    2) stats per waypoint go into 'run_pts' or 'bike_pts' table in 'tcx_activities' postgres database
    returns list of files that were parsed
        "run_stats": "(activity CHAR(14) PRIMARY KEY, activity_route CHAR(18) UNIQUE NOT NULL, activity_stats CHAR(18) UNIQUE NOT NULL, activity_type VARCHAR(5) NOT NULL, start TIMESTAMP UNIQUE NOT NULL, distance FLOAT NOT NULL, duration FLOAT NOT NULL, ascent FLOAT NOT NULL, avg_speed FLOAT NOT NULL, hr_avg FLOAT, hr_max FLOAT)",
    "bike_stats": "(activity CHAR(14) PRIMARY KEY, activity_route CHAR(18) UNIQUE NOT NULL, activity_stats CHAR(18) UNIQUE NOT NULL, activity_type VARCHAR(5) NOT NULL, start TIMESTAMP UNIQUE NOT NULL, distance FLOAT NOT NULL, duration FLOAT NOT NULL, ascent FLOAT NOT NULL, avg_speed FLOAT NOT NULL, hr_avg FLOAT, hr_max FLOAT)",
    '''
    ## for PST my start times need to be subtracted by six or seven hours depending on daylight savings
    run_files=[]
    bike_files=[]
    try:
        with psycopg2.connect(database = db, user = usr, password = pwd, host = host, port = port) as conn:
            with conn.cursor() as cur:
                for tf in tcx_files:
                    fname = os.path.basename(tf)
                    tcx_reader = TCXReader()
                    exercise = tcx_reader.read(tf)
                    if (exercise.activity_type == 'Running' and exercise.duration != None):
                        run_files.append(fname)
                        cur.execute('INSERT INTO run_stats (activity, activity_route, activity_stats, activity_type, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                                    [fname[:-4], fname[:-4]+'.gpx', fname, 'runs', exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                    elif (exercise.activity_type == 'Biking' and exercise.duration != None):
                        cur.execute('INSERT INTO bike_stats (activity, activity_route, activity_stats, activity_type, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING',
                                    [fname[:-4], fname[:-4]+'.gpx', fname, 'bikes', exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                        bike_files.append(fname)
                conn.commit()
        return (run_files, bike_files)
    except (Exception, psycopg2.Error) as error:
        print('Error inserting data into ', db, 'PostgreSQL', error)
    finally:
        if conn:
            conn.close()
            print('PostgreSQL connection to '+db+' is closed')        
    
def df_to_postgres(df, table_name, db, usr='postgres', pwd='', host='localhost', port=5432):
    try:
        with psycopg2.connect(database = db, user = usr, password = pwd, host = host, port = port) as conn:
            with conn.cursor() as cur:
                col_names = df.columns.to_list()
                for i in range(0 ,len(df)):
                    values = tuple(df[col][i] for col in col_names)
                    cur.execute('INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING'.format(table_name, ", ".join(col_names), str(values)[1:-1]))
                conn.commit()
    except (Exception, psycopg2.Error) as error:
        print('Error inserting data into ', table_name, 'PostgreSQL', error)
    finally:
        if conn:
            conn.close()
            print('PostgreSQL connection to '+table_name+' in '+db+' is closed')

def postgres_to_df(SQL_query, db, usr='postgres', pwd='', host='localhost', port=5432):
    try:
        with psycopg2.connect(database = db, user = usr, password = pwd, host = host, port = port) as conn:
            with conn.cursor() as cur:
                cur.execute(SQL_query)
                items = cur.fetchall()
                hits=[]
                for row in items:
                    hits.append(row)
                col_names = [desc[0] for desc in cur.description]
                hits_df=pd.DataFrame(hits, columns=col_names)
                if ('lat' in col_names and 'lon' in col_names):
                    hits_gdf = gpd.GeoDataFrame(hits_df, geometry=gpd.points_from_xy(hits_df.loc[:,'lon'],hits_df.loc[:,'lat'], crs='EPSG:4326'))
                    return hits_gdf
                else:
                    return hits_df

    except (Exception, psycopg2.Error) as error:
        print('Error while fetching data from PostgreSQL', error)
        conn.close()
        print('PostgreSQL connection to '+str(db)+'is closed')


################################
## RUNNING PLOTS
################################

def plot_heatmap(df, out_name, subset=1):
    #df = df.set_index('time').sort_index()
    df = df.iloc[::subset, :] ## subset every X points to keep file size down
    # for routes polylines
    all_routes_xy = []
    for k, v in df.groupby(['filename']):
        route_lats = v['lat'].to_list()
        route_lons = v['lon'].to_list()
        all_routes_xy.append(list(zip(route_lats, route_lons)))
    #df = df[start:end].dropna()
    heatmap = folium.Map(location=[np.mean(df['lat']), np.mean(df['lon'])], control_scale=False, zoom_start=12)
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = True, control = True, show = True).add_to(heatmap)
    cluster = plugins.HeatMap(data=[[la, lo] for la, lo in zip(df['lat'], df['lon'])], name='heatmap', min_opacity=0.15, max_zoom=10,  radius=9, blur=8)
    heatmap.add_child(cluster)
    fg = folium.FeatureGroup('routes')
    folium.PolyLine(locations=all_routes_xy, weight=0.9, opacity = 0.85, color = 'red', control = True, show = True).add_to(fg)
    fg.add_to(heatmap)
    folium.LayerControl().add_to(heatmap)
    heatmap.save(out_name)
    return heatmap

def plot_routemaps(run_df, bike_df, out_name):
    '''
    color running activities red and biking activities blue 
    '''
    # for routes polylines
    runs_xy = []
    for k, v in run_df.groupby(['filename']):
        route_lats = v['lat'].to_list()
        route_lons = v['lon'].to_list()
        runs_xy.append(list(zip(route_lats, route_lons)))
    bikes_xy = []
    for k, v in bike_df.groupby(['filename']):
        route_lats = v['lat'].to_list()
        route_lons = v['lon'].to_list()
        bikes_xy.append(list(zip(route_lats, route_lons)))
    ## create folium map 
    heatmap = folium.Map(location=[float((np.mean(bike_df['lat'])+np.mean(run_df['lat']))/2), float((np.mean(bike_df['lon'])+np.mean(run_df['lon']))/2)], 
                         control_scale=False, zoom_start=7)
    ## add basemap
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = True, control = True, show = True).add_to(heatmap)
    ## clusters 
    # cluster = plugins.HeatMap(data=[[la, lo] for la, lo in zip(run_df.lat, run_df.lon)], name='heatmap', min_opacity=0.15, max_zoom=10,  radius=9, blur=8)
    # heatmap.add_child(cluster)
    # cluster2 = plugins.HeatMap(data=[[la, lo] for la, lo in zip(bike_df.lat, bike_df.lon)], name='heatmap', min_opacity=0.15, max_zoom=10,  radius=9, blur=8)
    # heatmap.add_child(cluster2)
    ## lines
    fg1 = folium.FeatureGroup('running routes')
    fg2 = folium.FeatureGroup('biking routes')
    folium.PolyLine(locations=runs_xy, weight=0.9, opacity = 0.7, color = 'red', control = True, show = True).add_to(fg1)
    folium.PolyLine(locations=bikes_xy, weight=0.9, opacity = 0.7, color = 'blue', control = True, show = True).add_to(fg2)
    fg1.add_to(heatmap)
    fg2.add_to(heatmap)
    ## extra
    folium.LayerControl().add_to(heatmap)
    heatmap.save(out_name)
    return heatmap
    
def plot_3d(df, out_fi, subset=2):
    df = df.iloc[::subset, :] ## subset every X points to keep file size down
    route_3d = px.scatter_3d(df, x='lon', y='lat', z='ele', color='ele')
    route_3d.update_traces(marker={"size":1.0}, selector=dict(mode='markers'))
    route_3d.write_html(out_fi)
    return route_3d

def cal_heatmap(df, col_name, mpl_cmap, out_name):
    df = df[df['ascent'] < 20000]
    # Create a column containing the month
    df['month'] = pd.to_datetime(df['start']).dt.to_period('M')
    df['week'] = pd.to_datetime(df['start']).dt.to_period('W')
    df = df.reindex(sorted(df.columns), axis=1)
    df['Year'] = [int(str(i).split('-')[0]) for i in df['start'].astype(str)]
    df = df.sort_values('start')
    events = pd.Series([i for i in df[col_name]],
                       index=[i for i in  df['start'].astype('datetime64[ns]') - timedelta(hours=6)])
    cal_fig = calplot.calplot(events, suptitle='running '+col_name+' per day', cmap=mpl_cmap, colorbar=True, yearlabel_kws={'fontname':'sans-serif'})
    plt.savefig(out_name)
    return cal_fig[0]


def make_maps(route_df, date_df, hm_bounds, bounds, running_fig_dir, act_type, heatmap_cal_stats):
    ## i) create route heatmap (from .gpx files per activity type in archive directory)
    if (hm_bounds == 'nan' and len(route_df) > 0):
          plot_heatmap(df = route_df, out_name = os.path.join(running_fig_dir, 'HeatMap_'+act_type+'.html'), subset = 1) ## subset = remove every X waypoint to keep file size down      
    elif (hm_bounds != 'nan' and len(route_df) > 0):
        minlon, maxlon, minlat, maxlat = [float(i.replace(" ", "")) for i in str(hm_bounds).split(',')]
        heatmap_sub =  route_df[(route_df['lat'] >= minlat) & (route_df['lat'] <= maxlat) & (route_df['lon'] >= minlon) & (route_df['lon'] <= maxlon)]
        if len(heatmap_sub) > 0:
            plot_heatmap(df = heatmap_sub, out_name = os.path.join(running_fig_dir, 'HeatMap_'+act_type+'.html'), subset = 1) 
        else:
            print('Not creating RouteMap for '+act_type+'. No activities within those bounds')
    else:
        print('Not creating RouteMap for '+act_type+'. No activities within user input days')

    ## only create ii) 3D map for a square-ish subset of area (from .gpx files per activity type within bounds)
    if (len(bounds) > 0 and len(route_df) > 0):
        min_lon, max_lon, min_lat, max_lat = [float(i.replace(" ", "")) for i in str(bounds).split(',')]
        df_sub =  route_df[(route_df['lat'] >= min_lat) & (route_df['lat'] <= max_lat) & (route_df['lon'] >= min_lon) & (route_df['lon'] <= max_lon)]
        if len(df_sub) > 0:
            plot_3d(df = df_sub, out_fi = os.path.join(running_fig_dir, 'Route3D_'+act_type+'.html'), subset = 1) 
        else:
          print('Not creating 3D Map for '+act_type+'. No activities within those bounds')
    elif (len(bounds) > 0 and len(route_df) == 0):
          print('Not creating 3D Map for '+act_type+'. No activities within user input days')

    else:
        print('in https://geojson.io/: draw square over an area to map in 3D ')
        print('Provide a square-ish bounding box region -- min lon, max lon, min lat, max lat -- for the 3D map')

    ## iii) heatmap calendar (from .tcx files per activity type in archive directory)
    for heatmap_cal_stat in heatmap_cal_stats:
        if len(date_df) > 1:
            cal_heatmap(df = date_df,
                        col_name = heatmap_cal_stat.replace(" ", ""),
                        mpl_cmap = 'YlGn',
                        out_name = os.path.join(running_fig_dir, 'HeatCal_'+act_type+'_'+heatmap_cal_stat.replace(" ", "")+'.png'))        
                    
######################################################################

def main():
    start_time = time.time()


    ## parse user input params / instead of sourse config_file.sh
    df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'params.csv'))
    days_b4_today, hm_bounds, bounds, hcs, running_fig_dir, archive_dir = [str(i) for i in df['User Input']]
    heatmap_cal_stats = [str(i) for i in str(hcs).split(', ')]
    ## create archive and output figure directories if they don't exist
    if not os.path.exists(running_fig_dir):
        try:
            os.makedirs(running_fig_dir)
        except:
            print('Map output directory parameter in params.csv is not a valid filepath')
    if not os.path.exists(archive_dir):
        try:
            os.makedirs(archive_dir)
        except:
            print('Archive directory parameter in params.csv is not a valid filepath')

    ## if the first user input parameter can be an integer, download garmin activity files
    try:
        days_b4_today = int(days_b4_today)
        out_dir =  get_garmin(num_days = days_b4_today, 
                              project_dir = os.path.dirname(os.path.abspath(__file__)), 
                              file_types = ['.tcx', '.gpx'])
    ## or if it's a string, make that out_dir (a place where activity files to be parsed are located) 
    except:
        if os.path.exists(str(days_b4_today)):
            out_dir = str(days_b4_today)
            print('Parsing Strava .gpx files...')
        else:
            print('can not find filepath - relative to this script, activities.py')

    ## for CLOUD - google colab - workflow
    if (os.getcwd().startswith("/content/") and type(days_b4_today) == type(1)): 
        cloudlocal = 'cloud'
        run_df, bike_df = tcx_to_df(tcx_files = [os.path.join(out_dir, i) for i in sorted(os.listdir(out_dir)) if i.endswith('.tcx')])
        run_files = run_df['filename']
        bike_files = bike_df['filename']
    ## for LOCAL - mac / linux / windows - workflow
    else: 
        import gpxpy
        cloudlocal = 'local'
        postgres_db = 'activities'
        run_files, bike_files = tcx_to_postgres(tcx_files = [os.path.join(out_dir, i) for i in sorted(os.listdir(out_dir)) if i.endswith('.tcx')],  db = postgres_db)

    ## initialize list to put df's of running and biking routes 
    rb_dfs=[]        
    ## parse all activities that are running and/or biking - save files to csv 
    for act_type in ['bikes', 'runs']:
        if (act_type == 'runs' and len(run_files) > 0):
            parse_new = run_files
        elif (act_type == 'bikes' and len(bike_files) > 0):
            parse_new = bike_files
        else:
            print('no new '+act_type+' to parse')
            parse_new=[]
        if len(parse_new) >= 1:
            if cloudlocal == 'cloud':
                if act_type=='runs':
                    date_df = run_df
                elif act_type == 'bikes':
                    date_df = bike_df
                ## use tcx filenames to only parse those gpx files 
                date_df.to_csv(os.path.join(out_dir, "TCX_" + act_type + "_" + os.path.basename(out_dir) + ".csv"))
                route_df = gpx_to_df(files = [os.path.join(out_dir, i.replace('.tcx', '.gpx')) for i in parse_new] )
                route_df.to_csv(os.path.join(out_dir, "GPX_" + act_type + "_" + os.path.basename(out_dir) + ".csv"))
            elif cloudlocal == 'local':
                gpx_to_postgres(gpx_files = [os.path.join(out_dir, i.replace('.tcx', '.gpx')) for i in parse_new], 
                                table_name = 'route_'+act_type,  db = postgres_db)   
                ## select all activity_type route files in archive db
                route_df = postgres_to_df("SELECT * FROM route_"+act_type+";", db = postgres_db)
                ## select all activity_type date/stat files in archive db
                date_df = postgres_to_df("SELECT * FROM "+act_type[:-1]+"_stats", db = postgres_db)
                
            ## CREATE MAPS
            make_maps(route_df, date_df, hm_bounds, bounds, running_fig_dir, act_type, heatmap_cal_stats)
            rb_dfs.append(route_df)

        ## LOCAL + CLOUD: create combined bike+runs route heatmap if run + bikes both have activities within those days 
        if (len(rb_dfs) == 2 and(len(rb_dfs[0]) > 0 and len(rb_dfs[-1]) > 0)): 
            plot_routemaps(run_df = rb_dfs[0], bike_df = rb_dfs[-1], 
                           out_name = os.path.join(running_fig_dir, 'HeatMap_run_bike.html')) 

    ## LOCAL + CLOUD: ARCHIVE
    ## move all of the activity files themselves .gpx, .tcx in to 'archive' folder
    for file in os.listdir(out_dir):
        shutil.move(os.path.join(out_dir, file), os.path.join(archive_dir, file))
        ## delete today's directory
    if len(os.listdir(out_dir)) == 0:
        shutil.rmtree(out_dir)

    stop_time = time.time()
    time_min = (stop_time - start_time)/60
    print(str(days_b4_today), ' days took ', str(time_min), ' minutes')
######################################################################


if __name__ == "__main__":
    main()       
