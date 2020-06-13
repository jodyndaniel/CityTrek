from sklearn.neighbors import BallTree
from osgeo import gdal
import pyproj
import numpy
import matplotlib
import matplotlib.pyplot as plt
import osmnx  #
import geopandas  # needed get background data
import requests  # for making standard html requests
import json  # for parsing city of Toronto Police data
import pandas  # premier library for data organization
from pandas.io.json import json_normalize
import numpy  # for data manipulation
import networkx
import rasterstats
import rasterio
from osgeo import osr
from random import sample
from random import shuffle
from shapely.ops import nearest_points
import pickle

###################################################################################################
# Terrain Data #
###################################################################################################
# Need to pull in the terrain data so that I can add them to the road network nodes

filepath_TSC = r"C:/Users/jodyn/PycharmProjects/InsightDS/data/cleaned/TSC_2958.tif"
filepath_Hillshade = r"C:/Users/jodyn/PycharmProjects/InsightDS/data/cleaned/Hillshade_2958.tif"

# Open the file:
raster_TSC = gdal.Open(filepath_TSC)
raster_Hillshade = gdal.Open(filepath_Hillshade)
###################################################################################################
# Open Street Map #
###################################################################################################
# I need these data as the basis for the shortest path algorithm.
# I need a polygon of Toronto to define the limits of my demo

place_name = 'Old Toronto, Toronto, Golden Horseshoe, Ontario, Canada'

# Demo limits
city_GTA = osmnx.gdf_from_place(place_name)
osmnx.plot_shape(city_GTA)  # this looks okay

# Transporation data to later measure distance to feature for regression/random forest
selected_tags = {
    'highway': True}

stations_OldToronto = osmnx.pois_from_place(
    place=place_name, tags=selected_tags)

# need a UTM projection for distance to feature
features_location_names = [
    'crossing', 'give_way', 'stop',
    'traffic_signals', 'turning_loop', 'speed_camera']

# dropping some fetaures that I do not need
# some of them are polygons
stations_OldToronto_4326 = stations_OldToronto[stations_OldToronto.highway.isin(features_location_names)]
stations_OldToronto_4326 = stations_OldToronto_4326.iloc[:, numpy.r_[0, 1, 4]]
stations_OldToronto_4326.crs

# converting to other projections gives me inf
# also the gemteory tab is in espg 2958, but it read in as espg 4326
# I will need to covert this to a etract the x y data
# convert to a dataframe, then reproject
stations_OldToronto_4326['x'] = stations_OldToronto_4326['geometry'].x
stations_OldToronto_4326['y'] = stations_OldToronto_4326['geometry'].y
stations_OldToronto_4326 = stations_OldToronto_4326.reset_index(drop=True)
stations_OldToronto_4326 = pandas.DataFrame(stations_OldToronto_4326)


# I will need to create geaodataframes many times, so a function will be best to make the code cleaner
def create_gdf(df, Longitude, Latitude, projection):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[Longitude], df[Latitude]),
                                  crs=projection)

# converting the node data to a geopanda dataframe so that I can extract elevation data
stations_OldToronto_4326 = create_gdf(df=stations_OldToronto_4326,
                                      Latitude="y",
                                      Longitude="x",
                                      projection="EPSG:2958")
stations_OldToronto_2958 = create_gdf(df=stations_OldToronto_4326,
                                      Latitude="y",
                                      Longitude="x",
                                      projection="EPSG:2958")

stations_OldToronto_4326 = stations_OldToronto_4326.to_crs(epsg=4326)  # world.to_crs(epsg=3395) would also work

# I am interested in comparing the different street data available
# Loading : 1) walking paths, 2) all roads, 3) traffic roads
walk_path_GTA = osmnx.graph_from_place(place_name, network_type='walk')

# need to reproject to UTM for later analysis, but will work with all roads for my demo
walk_path_GTA_proj = osmnx.project_graph(walk_path_GTA)

# need edges and nodes to append data to
walk_nodes_proj, walk_edges_proj = osmnx.graph_to_gdfs(walk_path_GTA, nodes=True, edges=True)

# making sure its in NAD 1983 and save
walk_nodes_proj = walk_nodes_proj.to_crs({'init': 'epsg:2958'})

walk_nodes_proj['Index'] = walk_nodes_proj.index
# converting the node data to a geopanda dataframe so that I can extract elevation data
walk_node_id = pandas.DataFrame(walk_nodes_proj)
walk_nodes_gdf = create_gdf(df=walk_node_id,
                            Latitude="y",
                            Longitude="x",
                            projection="EPSG:4326")
walk_nodes_gdf = walk_nodes_gdf.to_crs(epsg=2958)  # world.to_crs(epsg=3395) would also work

walk_nodes_gdf.crs
##################################################################################################
# Crime Data #
###################################################################################################
# Because this is GEOJSON data, I could not use Beautifulsoup4
# So, the processes below is really to import and convert to
# a data frame format. Then, I can explore these data.
URL_Robbery = 'https://opendata.arcgis.com/datasets/9115accc55f24938b4eb573dd222c33b_0.geojson'
URL_Pedestrian = 'https://opendata.arcgis.com/datasets/1e8a71c533fb4b0aa522cf1b1236bee7_0.geojson'
page_robbery = requests.get(URL_Robbery).json()
page_pedestrian = requests.get(URL_Pedestrian).json()

pedestrian_json_df = pandas.json_normalize(page_pedestrian['features'])
# 4998 observations, 61 variables
# 2:7,11-12,19-20,24-25 (these are the columns I think I might want)
robbery_json_df = pandas.json_normalize(page_robbery['features'])
# 21543 observations, 30 variables

# Strange, but each column has a properties. prefix
# Looks wierd, so I am going to remove it
pedestrian_json_df.columns = pedestrian_json_df.columns.str.lstrip('properties.')
robbery_json_df.columns = robbery_json_df.columns.str.lstrip('properties.')
# focusing on features that I believe are predictive
pedestrian_df = pedestrian_json_df.iloc[:, numpy.r_[3,5,6, 10, 14,15, 19, 20]]
robbery_df = robbery_json_df.iloc[:, numpy.r_[1:3, 9, 10, 11, 12, 13, 14, 5, 8, 22, 23, 25, 26]]

# I will subset the data. For the robbery data, I only want the mugging data
# because this is most applicable for pedestrians
robbery_df_mugging = robbery_df.loc[robbery_df['ffence'] == 'Robbery - Mugging']  # 6847 rows
robbery_df_mugging_outside = robbery_df_mugging.loc[robbery_df_mugging['misetype'] == 'Outside']  # 5141 rows

# Some features are unablanced. I will remove accident locations with few observations (<10)
pedestrian_df_road = pedestrian_df.loc[pedestrian_df['ROAD_CLASS'].isin(['Major Arterial', 'Minor Arterial',
                                                                         'Collector', 'Local',
                                                                         'Expressway'])]  # 4986 rows

###################################################################################################
# Background Data #
# need as spatial data to measure distance to features later
robbery_points_gdf_2958 = create_gdf(df=robbery_df_mugging_outside,
                                     Latitude="Lat",
                                     Longitude="Long",
                                     projection="EPSG:4326")

robbery_points_gdf_4326 = create_gdf(df=robbery_df_mugging_outside,
                                     Latitude="Lat",
                                     Longitude="Long",
                                     projection="EPSG:4326")
robbery_points_gdf_2958 = robbery_points_gdf_2958.to_crs(
    {'init': 'epsg:2958'})  # world.to_crs(epsg=3395) would also work

robbery_points_gdf_2958.crs
robbery_points_gdf_2958.crs
# need consistency in labelling for future analyses
robbery_points_gdf_2958 = robbery_points_gdf_2958.rename(columns={"Long": "Longitude", "Lat": "Latitude"})
robbery_points_gdf_4326 = robbery_points_gdf_4326.rename(columns={"Long": "Longitude", "Lat": "Latitude"})

###################################################################################################
# Background Data #
pedestrian_points_gdf_2958 = create_gdf(df=pedestrian_df_road,
                                     Latitude="LATITUDE",
                                     Longitude="LONGITUDE",
                                     projection="EPSG:4326")

pedestrian_points_gdf_4326 = create_gdf(df=pedestrian_df_road,
                                        Latitude="LATITUDE",
                                        Longitude="LONGITUDE",
                                     projection="EPSG:4326")
pedestrian_points_gdf_2958 = pedestrian_points_gdf_2958.to_crs(
    {'init': 'epsg:2958'})  # world.to_crs(epsg=3395) would also work

pedestrian_points_gdf_2958.crs
pedestrian_points_gdf_4326.crs
# need consistency in labelling for future analyses
pedestrian_points_gdf_2958 = pedestrian_points_gdf_2958.rename(columns={"LONGITUDE": "Longitude", "LATITUDE": "Latitude"})
pedestrian_points_gdf_4326 = pedestrian_points_gdf_4326.rename(columns={"LONGITUDE": "Longitude", "LATITUDE": "Latitude"})
###################################################################################################
# it will be easier to re-run making the psedoabsences if I make a function
# considering that I might extend to the GTA and not just Old Toronto
def get_pseudoabsence_data(lat_min, lat_max, lon_min, lon_max, number_of_points):
    numpy.random.seed(13)
    pseudoabsence_latitudes = numpy.random.uniform(lat_min, lat_max, number_of_points)
    pseudoabsence_longitudes = numpy.random.uniform(lon_min, lon_max, number_of_points)
    pseudoabsences = pandas.DataFrame(numpy.transpose(numpy.array([pseudoabsence_longitudes, pseudoabsence_latitudes])),
                                      columns=['Longitude', 'Latitude'])
    return (pseudoabsences)
###################################################################################################
# some of the background points will fall in the lake, so multipling by 4 to make sure
# I have enough falling on land
robbery_backgroundpoints = get_pseudoabsence_data(lat_min=robbery_points_gdf_4326['Latitude'].min(),
                                                  lat_max=robbery_points_gdf_4326['Latitude'].max(),
                                                  lon_min=robbery_points_gdf_4326['Longitude'].min(),
                                                  lon_max=robbery_points_gdf_4326['Longitude'].max(),
                                                  number_of_points=len(robbery_points_gdf_4326) * 4)

# need to make spatial to extract proximity features
robbery_backgroundpoints_gdf_2958 = create_gdf(df=robbery_backgroundpoints,
                                               Latitude="Latitude",
                                               Longitude="Longitude",
                                               projection="EPSG:4326")
robbery_backgroundpoints_gdf_4326 = create_gdf(df=robbery_backgroundpoints,
                                               Latitude="Latitude",
                                               Longitude="Longitude",
                                               projection="EPSG:4326")

robbery_backgroundpoints_gdf_2958 = robbery_backgroundpoints_gdf_2958.to_crs(
    {'init': 'epsg:2958'})  # world.to_crs(epsg=3395) would also work

robbery_backgroundpoints_gdf_2958.crs
robbery_backgroundpoints_gdf_4326.crs
###################################################################################################
###################################################################################################
# some of the background points will fall in the lake, so multipling by 4 to make sure
# I have enough falling on land
pedestrian_backgroundpoints = get_pseudoabsence_data(lat_min=pedestrian_points_gdf_4326['Latitude'].min(),
                                                  lat_max=pedestrian_points_gdf_4326['Latitude'].max(),
                                                  lon_min=pedestrian_points_gdf_4326['Longitude'].min(),
                                                  lon_max=pedestrian_points_gdf_4326['Longitude'].max(),
                                                  number_of_points=len(pedestrian_points_gdf_4326) * 4)

# need to make spatial to extract proximity features
pedestrian_backgroundpoints_gdf_2958 = create_gdf(df=pedestrian_backgroundpoints,
                                               Latitude="Latitude",
                                               Longitude="Longitude",
                                               projection="EPSG:4326")
pedestrian_backgroundpoints_gdf_4326 = create_gdf(df=pedestrian_backgroundpoints,
                                               Latitude="Latitude",
                                               Longitude="Longitude",
                                               projection="EPSG:4326")

pedestrian_backgroundpoints_gdf_2958 = pedestrian_backgroundpoints_gdf_2958.to_crs(
    {'init': 'epsg:2958'})  # world.to_crs(epsg=3395) would also work

pedestrian_backgroundpoints_gdf_2958.crs
pedestrian_backgroundpoints_gdf_4326.crs
###################################################################################################
# Nodes & Edges Added For Terrain Data #
###################################################################################################
# add elevation to each of the nodes, using the google elevation API, then calculate edge grades
# will extract values of hillshade and TRI at each node in GIS

walk_nodes_gdf['TSC'] = rasterstats.point_query(walk_nodes_gdf, filepath_TSC,
                                                interpolate='nearest')
walk_nodes_gdf['Hillshade'] = rasterstats.point_query(walk_nodes_gdf, filepath_Hillshade,
                                                      interpolate='nearest')
###################################################################################################
# Distance To Features for RF/Logistic Regression #
###################################################################################################

# I am only interested in a few features for my demo, and the function below needs WGS 1984
stations_OldToronto_md = stations_OldToronto_4326.iloc[:, numpy.r_[0, 1, 2]]

stations_OldToronto_md['x'] = stations_OldToronto_md['geometry'].x
stations_OldToronto_md['y'] = stations_OldToronto_md['geometry'].y
###################################################################################################
def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = numpy.array(
        left_gdf[left_geom_col].apply(lambda geom: (geom.x * numpy.pi / 180, geom.y * numpy.pi / 180)).to_list())
    right_radians = numpy.array(
        right[right_geom_col].apply(lambda geom: (geom.x * numpy.pi / 180, geom.y * numpy.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['distance'] = dist * earth_radius

    return closest_points
###################################################################################################
# Find closest public transport stop for each building and get also the distance based on haversine distance
# Note: haversine distance which is implemented here is a bit slower than using e.g. 'euclidean' metric
# but useful as we get the distance between points in meters
for i in features_location_names:
    robbery_backgroundpoints_gdf_4326[i] = nearest_neighbor(robbery_backgroundpoints_gdf_4326,
                                                            stations_OldToronto_md.loc[
                                                                stations_OldToronto_md['highway'] == i],
                                                            return_dist=True)['distance']
    robbery_points_gdf_4326[i] = nearest_neighbor(robbery_points_gdf_4326,
                                                  stations_OldToronto_md.loc[
                                                      stations_OldToronto_md['highway'] == i],
                                                  return_dist=True)['distance']

# adding some psedo data based on the ranges of what are in the occurrences
special = list(set(list(robbery_points_gdf_4326)) - set(list(robbery_backgroundpoints_gdf_4326)))
for i in special:
    longer = list(robbery_points_gdf_4326[i].unique())
    call = numpy.repeat(longer,
                        len(robbery_backgroundpoints_gdf_4326) / len(longer) * 2)
    shuffle(call)
    robbery_backgroundpoints_gdf_4326[i] = call[0:len(robbery_backgroundpoints_gdf_4326)]

robbery_points_gdf_4326['Presence_Absence'] = numpy.repeat(1,
                                                           len(robbery_points_gdf_4326))
robbery_backgroundpoints_gdf_4326['Presence_Absence'] = numpy.repeat(0,
                                                                     len(robbery_backgroundpoints_gdf_4326))
# now we can combine both data sets and
robbery_model_data = robbery_points_gdf_4326.append(
    pandas.DataFrame(data=robbery_backgroundpoints_gdf_4326),
    ignore_index=True)
robbery_model_data_final = robbery_model_data.iloc[:, numpy.r_[0, 2:len(list(robbery_model_data))]]
robbery_model_data_final.Presence_Absence.value_counts()
# need some dummy variables for my categorical data
# One-hot encode the data using pandas get_dummies

robbery_model_data_dummies = pandas.get_dummies(robbery_model_data_final)
# NAs will give errors, so need to drop them
for i in list(robbery_model_data_dummies):
    robbery_model_data_dummies.dropna(subset=[i], inplace=True)

# do I need to balance data?
robbery_model_data_dummies.Presence_Absence.value_counts()
# yes
robbery_model_data_dummies_p = robbery_model_data_dummies.loc[
    robbery_model_data_dummies["Presence_Absence"] == 1]
robbery_model_data_dummies_a = robbery_model_data_dummies.loc[
    robbery_model_data_dummies["Presence_Absence"] == 0]
robbery_model_data_dummies_a = robbery_model_data_dummies_a.sample(n=len(robbery_model_data_dummies_p))

robbery_model_data_dummies_bal = robbery_model_data_dummies_p.append(
    pandas.DataFrame(data=robbery_model_data_dummies_a),
    ignore_index=True)

# Display the first 5 rows of the last 12 columns
##########################################################################################################
for i in features_location_names:
    pedestrian_backgroundpoints_gdf_4326[i] = nearest_neighbor(pedestrian_backgroundpoints_gdf_4326,
                                                            stations_OldToronto_md.loc[
                                                                stations_OldToronto_md['highway'] == i],
                                                            return_dist=True)['distance']
    pedestrian_points_gdf_4326[i] = nearest_neighbor(pedestrian_points_gdf_4326,
                                                  stations_OldToronto_md.loc[
                                                      stations_OldToronto_md['highway'] == i],
                                                  return_dist=True)['distance']

# adding some psedo data based on the ranges of what are in the occurrences
special = list(set(list(pedestrian_points_gdf_4326)) - set(list(pedestrian_backgroundpoints_gdf_4326)))
for i in special:
    longer = list(pedestrian_points_gdf_4326[i].unique())
    call = numpy.repeat(longer,
                        len(pedestrian_backgroundpoints_gdf_4326) / len(longer) * 2)
    shuffle(call)
    pedestrian_backgroundpoints_gdf_4326[i] = call[0:len(pedestrian_backgroundpoints_gdf_4326)]

pedestrian_points_gdf_4326['Presence_Absence'] = numpy.repeat(1,
                                                           len(pedestrian_points_gdf_4326))
pedestrian_backgroundpoints_gdf_4326['Presence_Absence'] = numpy.repeat(0,
                                                                     len(pedestrian_backgroundpoints_gdf_4326))
# now we can combine both data sets and
pedestrian_model_data = pedestrian_points_gdf_4326.append(
    pandas.DataFrame(data=pedestrian_backgroundpoints_gdf_4326),
    ignore_index=True)
pedestrian_model_data_final = pedestrian_model_data.iloc[:, numpy.r_[0, 2:len(list(pedestrian_model_data))]]
pedestrian_model_data_final.Presence_Absence.value_counts()
# need some dummy variables for my categorical data
# One-hot encode the data using pandas get_dummies

pedestrian_model_data_dummies = pandas.get_dummies(pedestrian_model_data_final)
# NAs will give errors, so need to drop them
for i in list(pedestrian_model_data_dummies):
    pedestrian_model_data_dummies.dropna(subset=[i], inplace=True)

# do I need to balance data?
pedestrian_model_data_dummies.Presence_Absence.value_counts()
# yes
pedestrian_model_data_dummies_p = pedestrian_model_data_dummies.loc[
    pedestrian_model_data_dummies["Presence_Absence"] == 1]
pedestrian_model_data_dummies_a = pedestrian_model_data_dummies.loc[
    pedestrian_model_data_dummies["Presence_Absence"] == 0]
pedestrian_model_data_dummies_a = pedestrian_model_data_dummies_a.sample(n=len(pedestrian_model_data_dummies_p))

pedestrian_model_data_dummies_bal = pedestrian_model_data_dummies_p.append(
    pandas.DataFrame(data=pedestrian_model_data_dummies_a),
    ignore_index=True)
##########################################################################################################
# Logistic Regression #
##########################################################################################################
                                             # Robbery Model #
# Labels are the values we want to predict
robbery_labels = robbery_model_data_dummies_bal['Presence_Absence']
# Remove the labels from the features
# axis 1 refers to the columns

#robbery_features = robbery_model_data_dummies_bal.iloc[:, numpy.r_[1:4, 9:14, 37:len(list(robbery_model_data_dummies_bal))]]
#robbery_model_names = list(robbery_model_data_dummies_bal.iloc[:, numpy.r_[1:4, 9:14, 37:len(list(robbery_model_data_dummies_bal))]])
robbery_features = robbery_model_data_dummies_bal.iloc[:, numpy.r_[1:4, 9:15]]
robbery_model_names = list(robbery_model_data_dummies_bal.iloc[:, numpy.r_[1:4, 9:15]])

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
robbery_train, robbery_test, robbery_presence_train, robbery_presence_test = train_test_split(
    robbery_features,
    robbery_labels, test_size=0.3, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
robbery_train = pandas.DataFrame(scaler.fit_transform(robbery_train),
                                   columns=robbery_model_names)

robbery_test = pandas.DataFrame(scaler.fit_transform(robbery_test),
                                  columns=robbery_model_names)

# Import the model we are using
from sklearn.linear_model import LogisticRegression

# Instantiate model with 1000 decision trees
robbery_model_LR = LogisticRegression()
robbery_model_LR.fit(robbery_train, robbery_presence_train)
robber_model_prediction = robbery_model_LR.predict(robbery_test)
# how did this model perform?
from sklearn.metrics import classification_report

print(classification_report(robbery_presence_test, robber_model_prediction))
robbery_model_LR.coef_
# 80 % if I keep Hood ID, 85 % accuracy if I drop it
import statsmodels.api as sm

# Getting the model coefficents from sklearn is bothersome
# So, I will use the statsmodels to view these
robbery_model_LR_SM = sm.Logit(list(robbery_presence_train), robbery_train)
robbery_model_LR_SM_2 = robbery_model_LR_SM.fit()
print(robbery_model_LR_SM_2.summary())
# 0.3754 if I keep Hood ID, 0.3533 is I drop it
# I think I should drop the dummy variables
# they don't really help model fit/predictions
#######################################################################################################
# Predict Probabilities for Data #
#######################################################################################################
# I am going to run this on the full dataset
# and select the features in this final model
robbery_features_final = robbery_model_data_dummies_bal.iloc[:, numpy.r_[1:4, 9:15]]
robbery_samples_final = robbery_model_data_dummies_bal['Presence_Absence']
robbery_features_final = pandas.DataFrame(scaler.fit_transform(robbery_features_final),
                                          columns=robbery_model_names)
# will add these data to the main dataframe
robbery_model_data_dummies_bal['Safe'] = pandas.DataFrame((robbery_model_LR.predict_proba(robbery_features_final)))[0]
robbery_model_data_dummies_bal['Robbed'] = pandas.DataFrame((robbery_model_LR.predict_proba(robbery_features_final)))[
    1]
robbery_model_nodes = robbery_model_data_dummies_bal.iloc[:, numpy.r_[6:8, 15, 54, 55]]
# need as a geopanda dataframe to append to nodes
robbery_model_nodes_4326 = create_gdf(robbery_model_nodes,
                                      Longitude='Longitude',
                                      Latitude='Latitude',
                                      projection="EPSG:4326")
###################################################################################################
###################################################################################################
# Labels are the values we want to predict
pedestrian_labels = pedestrian_model_data_dummies_bal['Presence_Absence']
# Remove the labels from the features
# axis 1 refers to the columns
pedestrian_features = pedestrian_model_data_dummies_bal.iloc[:, numpy.r_[0:1, 5:10,12:len(list(pedestrian_model_data_dummies_bal))]]
pedestrian_features_names = list(pedestrian_model_data_dummies_bal.iloc[:, numpy.r_[0:1, 5:10,12:len(list(pedestrian_model_data_dummies_bal))]])
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
pedestrian_train, pedestrian_test, pedestrian_presence_train, pedestrian_presence_test = train_test_split(
    pedestrian_features,pedestrian_labels, test_size=0.3, random_state=42)

scaler = StandardScaler()
pedestrian_train = pandas.DataFrame(scaler.fit_transform(pedestrian_train),
                                   columns=pedestrian_features_names)

pedestrian_test = pandas.DataFrame(scaler.fit_transform(pedestrian_test),
                                  columns=pedestrian_features_names)
# Import the model we are using
from sklearn.linear_model import LogisticRegression

# Instantiate model with 1000 decision trees
pedestrian_model_LR = LogisticRegression()
pedestrian_model_LR.fit(pedestrian_train, pedestrian_presence_train)
pedestrian_model_prediction = pedestrian_model_LR.predict(pedestrian_test)
# how did this model perform?
from sklearn.metrics import classification_report

print(classification_report(pedestrian_presence_test, pedestrian_model_prediction))
pedestrian_model_LR.coef_
# 94 % as accuracy - aite, but not great
import statsmodels.api as sm

# Getting the model coefficents from sklearn is bothersome
# So, I will use the statsmodels to view these
pedestrian_model_LR_SM = sm.Logit(list(pedestrian_presence_train), pedestrian_train,max_iter=7600)
pedestrian_model_LR_SM_2 = pedestrian_model_LR_SM.fit(max_iter=7600)
print(pedestrian_model_LR_SM_2.summary())
# Pseudo R-squ  - 0.7393
# I think I should drop the dummy variables
# they don't really help model fit/predictions

#######################################################################################################
# Predict Probabilities for Data #
#######################################################################################################
# I am going to run this on the full dataset
# and select the features in this final model
pedestrian_features_final = pandas.DataFrame(scaler.fit_transform(pedestrian_features),
                                          columns=pedestrian_features_names)
# will add these data to the main dataframe
pedestrian_model_data_dummies_bal['No_Collison'] = pandas.DataFrame((pedestrian_model_LR.predict_proba(pedestrian_features_final)))[0]
pedestrian_model_data_dummies_bal['Collison'] = pandas.DataFrame((pedestrian_model_LR.predict_proba(pedestrian_features_final)))[1]
pedestrian_model_nodes = pedestrian_model_data_dummies_bal.iloc[:, numpy.r_[2:4,11,34,35]]
# need as a geopanda dataframe to append to nodes
pedestrian_model_nodes_4326 = create_gdf(pedestrian_model_nodes,
                                      Longitude='Longitude',
                                      Latitude='Latitude',
                                      projection="EPSG:4326")

###################################################################################################
# Nodes & Edges Added For Crime #
###################################################################################################
# modifying a function from above to include export closest distance information
# from node to nearest robbery (full dataframe) versus just distance

def nearest_neighbor_2(left_gdf, right_gdf, return_dist=False):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = numpy.array(
        left_gdf[left_geom_col].apply(lambda geom: (geom.x * numpy.pi / 180, geom.y * numpy.pi / 180)).to_list())
    right_radians = numpy.array(
        right[right_geom_col].apply(lambda geom: (geom.x * numpy.pi / 180, geom.y * numpy.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    # closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        earth_radius = 6371000  # meters
        closest_points['Distance'] = dist * earth_radius

    return closest_points


# need path in wgs
walk_nodes_gdf = walk_nodes_gdf.to_crs({'init': 'epsg:4326'})
# saving as something else so that I can add to my main dataframe
walk_nodes_gdf_NB = nearest_neighbor_2(walk_nodes_gdf, robbery_model_nodes_4326,
                                             return_dist=True)
walk_nodes_gdf_NB_pd = nearest_neighbor_2(walk_nodes_gdf, pedestrian_model_nodes_4326,
                                             return_dist=True)
# adding index for future join to full node dataset
walk_nodes_gdf_NB.set_index(walk_nodes_gdf.index,inplace = True)
walk_nodes_gdf_NB_pd.set_index(walk_nodes_gdf.index,inplace = True)
# also adding model outputs at each nodes for edge weights later
walk_nodes_gdf_NB['TSC'] = list(walk_nodes_gdf['TSC'])
walk_nodes_gdf_NB['Hillshade'] = list(walk_nodes_gdf['Hillshade'])
walk_nodes_gdf_NB['Collison'] = list(walk_nodes_gdf_NB_pd['Collison'])
walk_nodes_gdf_NB['No_Collison'] = list(walk_nodes_gdf_NB_pd['No_Collison'])
###########################################################################
# now, I need to pickle these files to load later in the shortest path analyses
walk_nodes_gdf_NB.to_pickle("data/cleaned/walknodes.pkl")
walk_edges_proj.to_pickle("data/cleaned/edges.pkl")
walk_nodes_proj.to_pickle("data/cleaned/nodes.pkl")
with open("data/cleaned/path.p", 'wb') as f:
    pickle.dump(walk_path_GTA_proj,f)
