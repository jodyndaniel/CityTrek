# I am pulling data from 3 sources
# Open Street Maps - road network, attractions and restaurants data
# City of Toronto Police - robbery & pedestrian accident information
import requests # for making standard html requests
import json # for parsing city of Toronto Police data
import pandas # premier library for data organization
import osmnx # for importing data from Open Street Maps
from pandas.io.json import json_normalize
import numpy # for data manipulation 
import matplotlib
import geopandas
import pandas
import networkx
import matplotlib.pyplot
from shapely.geometry import Point, MultiPoint
from shapely.ops import nearest_points

################### Open Street Maps Data ##########################
# I need geospatial data from open street maps that I can use for
# shortest path algorithm.
# I need a polygon of Toronto to define the limits of my demo
place_name = 'Old Toronto, Toronto, Golden Horseshoe, Ontario, Canada'
city_GTA = osmnx.gdf_from_place(place_name)
osmnx.plot_shape(city_GTA) # this looks okay
selected_tags = {
    'amenity' : False,
    'landuse' : False,
     'highway' : True}

selected_tags = {
    'highway' : True}
features_location_names = ['bus_stop','traffic_signals' , 'crossing', 'residential', 'stop', 'path',
                           'footway','secondary', 'tertiary', 'cycleway']


stations_OldToronto = osmnx.pois_from_place(place = place_name, tags = selected_tags)


# I am interested in comparing the different street data available
# Loading : 1) walking paths, 2) all roads, 3) traffic roads
walk_path_GTA = osmnx.graph_from_place(place_name, network_type='walk')
all_path_GTA = osmnx.graph_from_place(place_name, network_type='all')
drive_path_GTA = osmnx.graph_from_place(place_name, network_type='drive')

# need to reproject to UTM for later analysis
all_path_GTA_proj = osmnx.project_graph(all_path_GTA)

################### Safety Data ##########################

# Because this is GEOJSON data, I could not use Beautifulsoup4
# So, the processes below is really to import and convert to 
# a data frame format. Then, I can explore these data.
URL_Robbery = 'https://opendata.arcgis.com/datasets/9115accc55f24938b4eb573dd222c33b_0.geojson'
URL_Pedestrian = 'https://opendata.arcgis.com/datasets/13c4200e013141ea8dd2c9d7af5d7e6c_0.geojson'
page_robbery = requests.get(URL_Robbery).json()
page_pedestrian = requests.get(URL_Pedestrian).json()

pedestrian_json_df = json_normalize(page_pedestrian["features"])
# 4998 observations, 61 variables
# 2:7,11-12,19-20,24-25 (these are the columns I think I might want)
robbery_json_df = json_normalize(page_robbery["features"])
# 21543 observations, 30 variables

# Strange, but each column has a properties. prefix
# Looks wierd, so I am going to remove it
pedestrian_json_df.columns = pedestrian_json_df.columns.str.lstrip('properties.')
robbery_json_df.columns = robbery_json_df.columns.str.lstrip('properties.')
# focusing on features that I believe are predictive
pedestrian_df = pedestrian_json_df.iloc[:, numpy.r_[1:6,10,11,18,19,20,22,23,25,26]]
robbery_df = robbery_json_df.iloc[:, numpy.r_[1:3,9,10,11,12,13,14,5,8,22,23,25,26]]


# Now, lets look at the number of robbery cases per year
robbery_df.dyear.value_counts().sort_index()
# making a plot for my demo
robbery_df.dyear.value_counts().sort_index().plot(kind='barh')
# Lets look at the types of robbery cases
robbery_df.ffence.value_counts()
robbery_df.ffence.value_counts().plot(kind='barh')
# Now, lets look at the number of pedestrian accidents  per year
pedestrian_df.YEAR.value_counts().sort_index()
# another plot, might use in my demo
pedestrian_df.YEAR.value_counts().sort_index().plot(kind='barh')

# I am wondering whether there is a difference in the relative frequency
# of accidents based on the road type
# If there is, I might need to focus my shortest path alogorithim
# on not just walking paths

pedestrian_df.ROAD_CLASS.value_counts() # most accidents occurred along main roads

# making another plot for my demo
pedestrian_df.ROAD_CLASS.value_counts().plot(kind='barh')

# I will subset the data. For the robbery data, I only want the mugging data
# because this is most applicable for pedestrians
robbery_df_mugging = robbery_df.loc[robbery_df['ffence'] == 'Robbery - Mugging'] # 6847 rows
robbery_df_mugging_outside = robbery_df_mugging.loc[robbery_df_mugging['misetype'] == 'Outside'] # 5141 rows

# Some features are unablanced. I will remove accident locations with few observations (<10)
pedestrian_df_road = pedestrian_df.loc[pedestrian_df['ROAD_CLASS'].isin(['Major Arterial','Minor Arterial',
                                                                      'Collector', 'Local',
                                                                      'Expressway'])] # 4986 rows

import geopandas # needed get background data
# will need to create geaodataframes many times, so a function will be best to make the code cleaner
def create_gdf(df, Longitude, Latitude, projection):
    return geopandas.GeoDataFrame(df,geometry = geopandas.points_from_xy(df[Longitude],df[Latitude]),
                                  crs = projection)

# need as spatial data for measure later features
robbery_points_gdf = create_gdf(df = robbery_df_mugging_outside,
                                Latitude = "Lat",
                                Longitude = "Long",
                                projection = "EPSG:4326")

# need consistency in labelling for future analyses
robbery_points_gdf = robbery_points_gdf.rename(columns={"Long": "Longitude", "Lat": "Latitude"})

# it will be easier to re-run making the psedoabsences if I make a function
# considering that I might extent to the GTA and not just Old Toronto
def get_pseudoabsence_data(lat_min, lat_max, lon_min, lon_max, number_of_points):
    numpy.random.seed(13)
    pseudoabsence_latitudes = numpy.random.uniform(lat_min, lat_max, number_of_points)
    pseudoabsence_longitudes = numpy.random.uniform(lon_min, lon_max, number_of_points)
    pseudoabsences = pandas.DataFrame(numpy.transpose(numpy.array([pseudoabsence_longitudes, pseudoabsence_latitudes])),
                                      columns=['Longitude', 'Latitude'])
    return (pseudoabsences)


robbery_backgroundpoints = get_pseudoabsence_data(lat_min = robbery_points_gdf['Latitude'].min(),
                                    lat_max = robbery_points_gdf['Latitude'].max(),
                                    lon_min = robbery_points_gdf['Longitude'].min(),
                                    lon_max = robbery_points_gdf['Longitude'].max(),
                                    number_of_points =  len(robbery_points_gdf)*4)



# need to make spatial to extract proximity features
robbery_backgroundpoints_gdf = create_gdf(df = robbery_backgroundpoints,
                                Latitude = "Latitude",
                                Longitude = "Longitude",
                                projection = "EPSG:4326")


import folium # for visualing maps in a cleaner format

darkmap = folium.Map([43.8379021, -79.1299286], tiles= "CartoDb dark_matter")

robbery_background = zip(robbery_backgroundpoints_gdf["Latitude"], robbery_backgroundpoints_gdf["Longitude"])
robbery_occurence = zip(robbery_points_gdf["Latitude"], robbery_points_gdf["Longitude"])
for location in robbery_occurence:
    folium.CircleMarker(location=location,
        color = "white", radius=1).add_to(darkmap)
for location in robbery_background:
    folium.CircleMarker(location=location,
        color = "red",   radius=1).add_to(darkmap)

darkmap.save('plot_data.html')
# Import the Folium interactive html file
from IPython.display.IFrame import HTML
HTML('<iframe src=plot_data.html width=700 height=450></iframe>')


# some data is outside
# makes some of the spatial queries running faster
import shapely.speedups
shapely.speedups.enable()

def distance_to_feature(gdf, feature, gdf_feature):

# which points are within?
robbery_bp_gdf_in = robbery_backgroundpoints_gdf.within(city_GTA.loc[0, 'geometry'])
robbery_p_gdf_in = robbery_points_gdf.within(city_GTA.loc[0, 'geometry'])
# only save points within
robbery_backgroundpoints_in_gdf = robbery_backgroundpoints_gdf.loc[robbery_bp_gdf_in]
robbery_points_in_gdf = robbery_points_gdf.loc[robbery_p_gdf_in]

robbery_points_gdf.within(city_GTA.loc[0, 'geometry'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
city_GTA.plot(ax=ax, facecolor='gray');
robbery_points_in_gdf.plot(ax=ax, color='blue', markersize=5);
robbery_backgroundpoints_in_gdf.plot(ax=ax, color='black', markersize=5)


for x in features_location_names:
    # 1 - create unary union
    location_data = stations_OldToronto.loc[(stations_OldToronto['highway'] == x)]
    dest_unary = location_data["geometry"].unary_union
    # 2 - find closest point
    nearest_geom = nearest_points(row[col], dest_unary)
    # 3 - Find the corresponding geom
    match_geom = destination.loc[destination.geometry
                                 == nearest_geom[1]]


Test = nearest_points(robbery_points_in_gdf["geometry"].unary_union,
               robbery_backgroundpoints_in_gdf["geometry"].unary_union)

match_geom = robbery_points_in_gdf.loc[robbery_points_in_gdf.geometry
                                 == Test[1]]