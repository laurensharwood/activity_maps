#!/usr/bin/ python

"""
in terminal: 
export EMAIL=your garmin username/email 
export PASSWORD=your garmin pwd
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
import gpxpy
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
    """Initialize Garmin API with your credentials."""
    try:
        # if not email or not password:
        #     email, password = get_credentials()
       ## print(f"Trying to login to Garmin Connect using token data from ...\n'{tokenstore}'" )
        garmin = Garmin()
        garmin.login(tokenstore)
        garth.save("~/.garth")

    except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError):
        # Session is expired. You'll need to log in again
       ## print(  "Login tokens not present, login with your Garmin Connect credentials to generate them.\n"  f"They will be stored in '{tokenstore}' for future use.\n")
        email, password = get_credentials()
        garmin = Garmin(email, password)
        garmin.login()
        garth.save("~/.garth")
        try:
            garmin = Garmin(email, password)
            garmin.login()
            garth.save("~/.garth")

            # Save tokens for next login
            garmin.garth.dump(tokenstore)

        except (FileNotFoundError, GarthHTTPError, GarminConnectAuthenticationError, requests.exceptions.HTTPError) as err:
            logger.error(err)
            return None

    return garmin

######################################################################

def switch(api, option, out_dir, startdate, today):
    """Run selected API call."""
    if api:
        activitytype=""
        activities = api.get_activities_by_date(startdate.isoformat(), today.isoformat(), activitytype )
        for activity in activities:
            activity_id = activity["activityId"]
            activity_name = activity["activityName"]
            activity_start = activity["startTimeLocal"].replace(" ", "", -1).replace(":", "", -1).replace("-", "", -1)            
            if option==".gpx":
                gpx_data = api.download_activity( activity_id, dl_fmt=api.ActivityDownloadFormat.GPX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.gpx"), "wb") as fb:
                    fb.write(gpx_data)
            elif option==".tcx":
                tcx_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.TCX)
                with open(os.path.join(out_dir, f"{str(activity_start)}.tcx"), "wb") as fb:
                    fb.write(tcx_data)
            elif option==".zip":
                zip_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.ORIGINAL)
                with open(os.path.join(out_dir, f"{str(activity_start)}.zip"), "wb") as fb:
                    fb.write(zip_data)
            elif option== ".csv":
                csv_data = api.download_activity(activity_id, dl_fmt=api.ActivityDownloadFormat.CSV)
                with open(os.path.join(out_dir, f"{str(activity_start)}.csv"), "wb") as fb:
                    fb.write(csv_data)

def get_garmin(num_days, project_dir, file_types):
    '''
    navigate into project directory (location of script) & run from there
    file_types: all options = [".tcx", ".gpx", ".csv", ".zip"]
    saves activity files into a new folder named today's date (YYYYMMDD) located in the script/project directory
    '''
    ## keep python script in the project directory with archive & out folders
    ## Load environment variables with Garmin credentials
    # email = os.getenv("GARMINEMAIL")
    # password = os.getenv("GARMINPWD")
    # tokenstore = os.getenv("GARMINTOKENS") or "~/.garminconnect"
    api = None

    today = date.today()
    startdate = today - timedelta(days=int(num_days))
    YYYYMMDD = today.strftime("%Y%m%d")
    out_dir = os.path.join(project_dir, YYYYMMDD)
    print(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    archive_dir = os.path.join(project_dir, "archive")
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    running_fig_dir = os.path.join(project_dir, "out")
    if not os.path.exists(running_fig_dir):
        os.makedirs(running_fig_dir)

    garth.save("~/.garth")

    ## 1) download from garmin
    for option in file_types:
        menu_options = {
            option: f"Download activities data by date from '{startdate.isoformat()}' to '{today.isoformat()}'",
            "q": "Exit"}
        if not api:
            email, password = get_credentials()
            tokenstore = "~/.garminconnect"
            api = init_api(email, password, tokenstore)
        if api:
            switch(api, option, out_dir, startdate, today)
        else:
            api = init_api(email, password, tokenstore)
    if len(os.listdir(out_dir)) == 0:
        os.rmdir(out_dir)

    return out_dir

################################
## TCX / GPX ACTIVITY PARSING
################################
def parse_tcx(data_dir):
    '''
    parse all TCX files in data directory -- make directory name the date of last activity -- writes csv with that date
    function is quick so okay to reparse TCX files -- for next run, add new files to folder
    '''
    date = os.path.basename(data_dir)
    tcx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".tcx")]
    print(tcx_files)
    df = pd.DataFrame(columns=['file', 'date', 'duration', 'distance', 'ascent', 'hr_max', 'cadence', 'avg_speed'])
    df_bike = pd.DataFrame(columns=['file', 'date', 'duration', 'distance', 'ascent', 'hr_max', 'avg_speed'])
    for tf in tcx_files:
        file = open(os.path.join(data_dir, tf), 'r')
        tcx_reader = TCXReader()
        TCXTrackPoint = tcx_reader.read(os.path.join(data_dir, tf))
        if TCXTrackPoint.activity_type == "Running":
            df.loc[len(df.index)] = [file.name, TCXTrackPoint.start_time, TCXTrackPoint.duration,
                                     TCXTrackPoint.distance, TCXTrackPoint.ascent, TCXTrackPoint.hr_max,
                                     TCXTrackPoint.tpx_ext_stats.get('RunCadence'), TCXTrackPoint.avg_speed]
        elif TCXTrackPoint.activity_type == "Biking":
            df_bike.loc[len(df_bike.index)] = [file.name, TCXTrackPoint.start_time, TCXTrackPoint.duration,
                                               TCXTrackPoint.distance, TCXTrackPoint.ascent, TCXTrackPoint.hr_max,
                                               TCXTrackPoint.avg_speed]
    ## clean running activity attributes
    df['file'] = [os.path.basename(i) for i in df['file']]
    df.set_index('file', inplace=True)
    df['name'] = [os.path.basename(f) for f in df.index]
    df['miles'] = [f / 1609.34 for f in df['distance']]
    df['seconds'] = df['duration']
    df['minutes'] = [f / 60 for f in df['seconds']]
    df['meters'] = df['distance']
    df['avg_mph'] = df['miles'] / df['minutes'] * 60
    df['min_per_mi'] = df['minutes'] / df['miles']
    df['cad_avg'] = [str(i).split(":")[-1].replace("}", "").replace(" ", "") for i in df.cadence]
    df['cad_max'] = [int(str(i).split(": ")[-3].split(", ")[0]) if len(str(i).split(": ")) == 4 else 0 for i in
                     df['cadence']]
    df['PST'] = df['date'].astype('datetime64[ns]') - timedelta(hours=6)
    df.sort_values('PST').dropna().to_csv(os.path.join(data_dir, 'runTCX_' + date + '.csv'))
    df['PST'] = [str(i) for i in df['PST']]
    return df, df_bike

def parse_gpx(data_dir):
    date = os.path.basename(data_dir)
    pts_df = pd.DataFrame(columns=['file', 'lat', 'lon', 'ele', 'speed', 'time'])
    gpx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".gpx")]
    for fi in gpx_files:
        gpx_file = open(os.path.join(data_dir, fi), 'r')
        gpx = gpxpy.parse(gpx_file, version="1.1")
        for track in gpx.tracks:
            for seg in track.segments:
                for point_no, pt in enumerate(seg.points):
                    if pt.speed != None:
                        speed = pt
                    elif point_no > 0:
                        speed = pt.speed_between(seg.points[point_no - 1])
                    elif point_no == 0:
                        speed = 0
                    pts_df.loc[len(pts_df.index)] = [fi, pt.latitude, pt.longitude, pt.elevation, speed, pt.time]
    pts_df['PST'] = pts_df['time'] - timedelta(hours=6)
    pts_df['PST'] = [str(i).split("+")[0] for i in pts_df['PST']]
    # pts_df.set_index('file', inplace=True)
    new_pt_csv = os.path.join(data_dir, "allGPX_" + date + ".csv")
    pts_df.sort_values('time').to_csv(new_pt_csv)
    return pts_df, new_pt_csv



def gpx_to_postgres(data_dir, table_name, db="garmin_activities"):
    '''
    data_dir  = input directory to put GPX waypoints in
    db = database
    table_name = table in db that gpx waypoints will be added to (should already exist in db)
        CREATE TABLE gpx_runs (filename CHAR(18) NOT NULL, date TIMESTAMP NOT NULL, lat FLOAT NOT NULL, lon FLOAT NOT NULL, ele FLOAT NOT NULL, speed FLOAT NOT NULL);
    returns list of files that were parsed
    '''
    gpx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".gpx")]
    try:
        conn = psycopg2.connect(database=db, user="postgres", password="", host="localhost", port=5432)
        cur = conn.cursor()
        for fi in gpx_files:
            gpx_file = open(os.path.join(data_dir, fi), 'r')
            gpx = gpxpy.parse(gpx_file, version="1.1")
            for track in gpx.tracks:
                for seg in track.segments:
                    for point_no, pt in enumerate(seg.points):
                        if pt.speed != None:
                            speed = pt
                        elif point_no > 0:
                            speed = pt.speed_between(seg.points[point_no - 1])
                        elif point_no == 0:
                            speed = 0
                        cur.execute("INSERT INTO "+table_name+" (filename, date, lat, lon, ele, speed) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                                    [fi, pt.time, pt.latitude, pt.longitude, pt.elevation, speed])
        conn.commit()
        print("Records created successfully")
        conn.close()
        return gpx_files

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")

def tcx_to_postgres(data_dir, db='garmin_activities'):
    '''
    data_dir  = input directory to parse all TCX files in
    1) adds basic activity stats to 'run_stats' or 'bike_stats' table in 'tcx_activities' postgres database
    2) stats per waypoint go into 'run_pts' or 'bike_pts' table in 'tcx_activities' postgres database
    returns list of files that were parsed
    '''

    ## for PST my start times need to be subtracted by six or seven hours depending on daylight savings
    subtr_hrs = 6
    tcx_files = [i for i in sorted(os.listdir(data_dir)) if i.endswith(".tcx")]
    try:
        conn = psycopg2.connect(database=db, user="postgres", password="", host="localhost", port=5432)
        cur = conn.cursor()
        for tf in tcx_files:
            file = open(os.path.join(data_dir, tf), "r")
            tcx_reader = TCXReader()
            exercise = tcx_reader.read(os.path.join(data_dir, tf))
            if (exercise.activity_type == "Running" and exercise.duration != None):
                cur.execute("INSERT INTO run_stats (filename, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                            [tf, exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                for pt_info in exercise.trackpoints:
                    cur.execute("INSERT INTO run_pts (date, filename, lon, lat, distance, elevation, hr, cadence) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                                [pt_info.time - timedelta(hours=subtr_hrs), tf, pt_info.longitude, pt_info.latitude, pt_info.distance,  pt_info.elevation, pt_info.hr_value, pt_info.cadence])
            elif (exercise.activity_type == "Biking" and exercise.duration != None):
                cur.execute("INSERT INTO bike_stats (filename, start, distance, duration, ascent, avg_speed, hr_avg, hr_max) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                            [tf, exercise.start_time - timedelta(hours=subtr_hrs), exercise.distance, exercise.duration, exercise.ascent, exercise.avg_speed, exercise.hr_avg, exercise.hr_max])
                for pt_info in exercise.trackpoints:
                    cur.execute("INSERT INTO bike_pts (date, filename, lon, lat, distance, elevation, hr) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                                [pt_info.time - timedelta(hours=subtr_hrs), tf, pt_info.longitude, pt_info.latitude, pt_info.distance,  pt_info.elevation, pt_info.hr_value])
        conn.commit()
        print("Records created successfully")
        conn.close()
        return tcx_files

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")


def postgres_to_df(SQL_query, db, user="postgres", pwd="", host="localhost", port=5432):
    try:
        conn = psycopg2.connect(database=db, user="postgres", password=pwd, host=host, port=port)
        cur = conn.cursor()
        cur.execute(SQL_query)
        items = cur.fetchall()
        hits=[]
        for row in items:
            hits.append(row)
        col_names = [desc[0] for desc in cur.description]
        hits_df=pd.DataFrame(hits, columns=col_names)
        if ("lat" in col_names and "lon" in col_names):
            hits_gdf = gpd.GeoDataFrame(hits_df, geometry=gpd.points_from_xy(hits_df.loc[:,'lon'],hits_df.loc[:,'lat'], crs="EPSG:4326"))
            return hits_gdf
        else:
            return hits_df

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        if conn:
            cur.close()
            conn.close()
            print("PostgreSQL connection is closed")


################################
## RUNNING PLOTS
################################

def plot_heatmap(df, out_name):
    #df = df.set_index("time").sort_index()
    df = df.iloc[::1, :] ## subset every 1 points to keep file size down
    # for routes polylines
    all_routes_xy = []
    for k, v in df.groupby(['file']):
        route_lats = v.lat.to_list()
        route_lons = v.lon.to_list()
        all_routes_xy.append(list(zip(route_lats, route_lons)))
    #df = df[start:end].dropna()
    heatmap = folium.Map(location=[np.mean(df.lat), np.mean(df.lon)], control_scale=False, zoom_start=12)
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = True, control = True, show = True).add_to(heatmap)
    cluster = plugins.HeatMap(data=[[la, lo] for la, lo in zip(df.lat, df.lon)], name="heatmap", min_opacity=0.15, max_zoom=10,  radius=9, blur=8)
    heatmap.add_child(cluster)
    fg = folium.FeatureGroup("routes")
    folium.PolyLine(locations=all_routes_xy, weight=1.0, opacity = 1.0, color='red', control = True, show = True).add_to(fg)
    fg.add_to(heatmap)
    folium.LayerControl().add_to(heatmap)
    heatmap.save(out_name)
    return heatmap

def plot_3d(df, out_fi):
    route_3d = px.scatter_3d(df, x='lon', y='lat', z='ele', color='ele')
    route_3d.update_traces(marker=dict(size=2.0), selector=dict(mode='markers'))
    route_3d.write_html(out_fi)
    return route_3d


def cal_heatmap(df, col_name, mpl_cmap, out_name):
    df = df[df.ascent < 20000]
    df['ascent_ft'] = [float(i)*float(3.28084) for i in df.ascent.to_list()]
    df['ascent_m'] = np.where( df['ascent'] >= 2000, 2000, df['ascent'])
    # Create a column containing the month
    df['month'] = pd.to_datetime(df['PST']).dt.to_period('M')
    df['week'] = pd.to_datetime(df['PST']).dt.to_period('W')
    df = df.reindex(sorted(df.columns), axis=1)
    df['Year'] = [int(str(i).split("-")[0]) for i in df.PST]
    df = df.sort_values('PST')
    events = pd.Series([i for i in df[col_name]],
                       index=[i for i in df['PST'].astype('datetime64[ns]')])
    cal_fig = calplot.calplot(events, suptitle="running "+col_name+" per day", cmap=mpl_cmap, colorbar=True, yearlabel_kws={'fontname':'sans-serif'})
    plt.savefig(out_name)
    return cal_fig[0]


######################################################################

def main():
    ## parse user input params / instead of sourse config_file.sh
    df = pd.read_csv("RUNfile.csv")
    days_b4_today = int(df.iloc[:,1][0])
    act_type = str(df.iloc[:,2][0])
    min_lon, max_lon, min_lat, max_lat = [float(i) for i in str(df.iloc[:,3][0]).split(", ")]
    heatmap_cal_stats = [i for i in df.iloc[:,4]]

    ## create archive and outut figure directories if they don't exist
    archive_dir = os.path.join(os.getcwd(), "garmin_archive")
    if not os.path.exists(archive_dir):
        os.makedirs(archive_dir)
    running_fig_dir = os.path.join(archive_dir, "figs")
    if not os.path.exists(running_fig_dir):
        os.makedirs(running_fig_dir)
    ## download garmin activity files
    out_dir =  get_garmin(num_days = days_b4_today, project_dir = os.getcwd(), file_types = [".tcx", ".gpx"])

    ## parse tcx files
    tcx_to_postgres(data_dir, db = 'garmin_activities')
    ## parse gpx files: save df for running and biking activities separately
    gpx_to_postgres(data_dir, db = 'garmin_activities')

    ## move all of the activity files themselves .gpx, .tcx in to 'archive' folder
    for file in os.listdir(out_dir):
        shutil.move(os.path.join(out_dir, file), os.path.join(archive_dir, file))
        ## delete today's directory
    if len(os.listdir(out_dir)) == 0:
        shutil.rmtree(out_dir)

    ## i) route heatmap
    full = postgres_to_df("SELECT * FROM gpx_runs WHERE lat >= "+str(min_lat)+" AND lat <= "+str(max_lat)+" AND lon > "+str(min_lon)+" AND lon < "+str(max_lon)+";",
    db = "garmin_activities", user = "postgres", pwd = "", host = "localhost", port = 5432)
    plot_heatmap(full, os.path.join(running_fig_dir, key[:-3]+"_heatMap.html"))

    ## ii) 3D routes
    ## sub = full[(full['lat'] >= min_lat) & (full['lat'] <= max_lat) & (full['lon'] >= min_lon) & (full['lon'] <= max_lon)]
    plot_3d(df = full, out_fi = os.path.join(running_fig_dir, key[:-3]+"_route3D.html"))
    ## iii) heatmap calendar
    cal_heatmap(df = postgres_to_df("SELECT * FROM gpx_"+act_type+"s;", db = "garmin_activities", user="postgres", pwd="", host="localhost", port=5432),
                col_name = heatmap_cal_stat,
                mpl_cmap = 'YlGn',
                out_name = os.path.join(running_fig_dir, key[:-3]+'_heatCal_'+heatmap_cal_stat+'.png'))

######################################################################


if __name__ == '__main__':
    main()       
