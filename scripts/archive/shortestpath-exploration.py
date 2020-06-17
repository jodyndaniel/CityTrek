import requests  # for making standard html requests
import json  # for parsing city of Toronto Police data
import pandas  # premier library for data organization
import osmnx  # for importing data from Open Street Maps
from pandas.io.json import json_normalize
import numpy  # for data manipulation
import matplotlib
import geopandas
import pandas
import networkx
import matplotlib.pyplot
import folium
import pickle
from shapely.geometry import Point, LineString, shape
from folium import IFrame

#####################################################################################################
############################## LOADING DATA ########################################################
walknodes = pandas.read_pickle('data/cleaned/walknodes.pkl')
G_nodes = pandas.read_pickle('data/cleaned/nodes.pkl')
G_edges = pandas.read_pickle('data/cleaned/edges.pkl')
with open("data/cleaned/path.p", 'rb') as f:
    G_walk = pickle.load(f)

# Modify gdf variables
G_nodes.drop(columns=['ref', 'highway', 'osmid', 'geometry'], inplace=True)
G_edges.drop(
    columns=['maxspeed', 'osmid', 'tunnel', 'ref', 'name', 'service', 'junction', 'bridge', 'access', 'area',
             'geometry', 'oneway', 'lanes'], inplace=True)
# removing irrelvant columns before merging
walknodes.drop(columns=['geometry', 'Longitude', 'Latitude', 'Presence_Absence',
                        'Distance'], inplace=True)

# now adding the model results to the nodes
G_nodes = G_nodes.join(walknodes)

# find the edge weights based on these model values
G_edges['Rob'] = [int(100 * (G_nodes.loc[u]['Robbed'] + G_nodes.loc[v]['Robbed']))
                       for u, v in zip(G_edges['u'], G_edges['v'])]
G_edges['Collison'] = [int(100 * (G_nodes.loc[u]['Collison'] + G_nodes.loc[v]['Collison']))
                       for u, v in zip(G_edges['u'], G_edges['v'])]
G_edges['Hillshade'] = [int(100 * (G_nodes.loc[u]['Hillshade'] + G_nodes.loc[v]['Hillshade']))
                       for u, v in zip(G_edges['u'], G_edges['v'])]
G_edges['TSC'] = [int(0.5 * (G_nodes.loc[u]['TSC'] + G_nodes.loc[v]['TSC']))
                       for u, v in zip(G_edges['u'], G_edges['v'])]

Robs = {}
Collisons = {}
Hillshades = {}
TSCs = {}

# Set each edge's tree weight as the average of the tree weights of the edge's vertices
for row in G_edges.itertuples():
    u = getattr(row, 'u')
    v = getattr(row, 'v')
    key = getattr(row, 'key')
    Rob = getattr(row, 'Rob')
    Collison = getattr(row, 'Collison')
    Hillshade = getattr(row, 'Hillshade')
    TSC = getattr(row, 'TSC')

    Robs[(u, v, key)] = Rob
    Collisons[(u, v, key)] = Collison
    Hillshades[(u, v, key)] = Hillshade
    TSCs[(u, v, key)] = TSC

networkx.set_edge_attributes(G_walk, Robs, 'Rob')
networkx.set_edge_attributes(G_walk, Collisons, 'Collison')
networkx.set_edge_attributes(G_walk, Hillshades, 'Hillshade')
networkx.set_edge_attributes(G_walk, TSCs, 'TSC')

##################################################################################################
# now, I need to pickle these files to load in the app later
G_edges.to_pickle("data/cleaned/edges.pkl")
G_nodes.to_pickle("data/cleaned/nodes.pkl")
with open("data/cleaned/path.p", 'wb') as f:
    pickle.dump(G_walk,f)
##################################################################################################
G_nodes = pandas.read_pickle('data/cleaned/nodes.pkl')
G_edges = pandas.read_pickle('data/cleaned/edges.pkl')
with open("data/cleaned/path.p", 'rb') as f:
    G_walk = pickle.load(f)
##################################################################################################
# function to create geopandas
def create_gdf(df, Longitude, Latitude, projection):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[Longitude], df[Latitude]),
                                  crs=projection)
##################################################################################################

orig_xy = osmnx.utils_geo.geocode("CN Tower, Toronto, Canada")
target_xy = osmnx.utils_geo.geocode("Casa Loma, Toronto, Canada")

try:
    osmnx.utils_geo.geocode("Kitchener City Hall, Toronto, Canada")
except AssertionError as error:
    print(error)
    print('function was not executed')

try:
    osmnx.utils_geo.geocode("CN Tower, Toronto, Canada")
except:
    pass

try:
    orig_xy = osmnx.geocode("Kitchener City Hall, Toronto, Canada")
except:
    orig_xy = " "

if not osmnx.utils_geo.geocode("Kitchener City Hall, Toronto, Canada"):
    return



orig_xy_form = ('Kitchener City Hall')
target_xy_form = ('Casa Loma')
orig_xy_geo = "%s, Toronto, Canada" % orig_xy_form
target_xy_geo = "%s, Toronto, Canada" % target_xy_form
try:
    orig_xy = osmnx.geocode(orig_xy_geo)
except:
    orig_xy = ''
try:
    target_xy = osmnx.geocode(target_xy_geo)
except:
    target_xy = ''

if  (orig_xy or target_xy):
        start_coords = (43.7112075, -79.4762563)
        print(start_coords)

# need in location UTMs to find nodes
start_stop = pandas.DataFrame([orig_xy, target_xy],
                              columns=['y', 'x'])
start_stop = create_gdf(df=start_stop,
                        Latitude="y",
                        Longitude="x",
                        projection="EPSG:4326").to_crs(epsg=2958)

# extracting in UTMs
orig_xy = (start_stop.geometry.y[0], start_stop.geometry.x[0])
target_xy = (start_stop.geometry.y[1], start_stop.geometry.x[1])

# Find the node in the graph that is closest to the origin point (here, we want to get the node id)
orig_node = osmnx.get_nearest_node(G_walk, orig_xy, method='euclidean')
# Find the node in the graph that is closest to the target point (here, we want to get the node id)
target_node = osmnx.get_nearest_node(G_walk, target_xy, method='euclidean')
##################################################################################################
# Calculate new edge weights
Collisons = {}
Hillshades = {}
Robs = {}
TSCs = {}
lengths = {}
for row in G_edges.itertuples():
    u = getattr(row, 'u')
    v = getattr(row, 'v')
    key = getattr(row, 'key')
    Collison = getattr(row, 'Collison')
    Hillshade = getattr(row, 'Hillshade')
    Rob = getattr(row, 'Rob')
    TSC = getattr(row, 'TSC')
    length = getattr(row, 'length')

    Collisons[(u, v, key)] = Collison
    Hillshades[(u, v, key)] = Hillshade
    Robs[(u, v, key)] = Rob
    TSCs[(u, v, key)] = TSC
    lengths[(u, v, key)] = length

# Optimized attribute is a weighted combo of normal length, tree counts, and road safety.
# Larger value is worse

optimized = {}
for key in lengths.keys():
    temp = (lengths[key])
    temp += (7 * (Robs[key]))
    temp += (5 * (TSCs[key]))
    temp += (6 * (Collisons[key]))
    temp += (1/40 * (Hillshades[key]))# max 50300
    optimized[key] = temp


optimized = {}
for key in lengths.keys():
    temp = int(lengths[key])
    temp += int(7 * (Robs[key]))
    temp += int(5 * (TSCs[key]))
    temp += int(6 * (Collisons[key]))
    temp += int(1/40 * (Hillshades[key]))# max 50300
    optimized[key] = temp

networkx.set_edge_attributes(G_walk, optimized, 'optimized')
# Path of nodes
optimized_route = networkx.shortest_path(G_walk, orig_node, target_node, weight='optimized')
# Path of nodes
shortest_route = networkx.shortest_path(G_walk, orig_node, target_node, weight='length')
# Get the nodes along the routes path
shortest_route_nodes = G_nodes.loc[shortest_route]
optimized_route_nodes = G_nodes.loc[optimized_route]
# converting to linestring in UTM to measure distance of routes
shortest_route_nodes_utm = create_gdf(df=shortest_route_nodes,
                                      Latitude="y",
                                      Longitude="x",
                                      projection="EPSG:4326").to_crs(epsg=2958)

shortest_route_nodes_utm = LineString(list(shortest_route_nodes_utm.geometry.values))

optimized_route_nodes_utm = create_gdf(df=optimized_route_nodes,
                                      Latitude="y",
                                      Longitude="x",
                                      projection="EPSG:4326").to_crs(epsg=2958)

optimized_route_nodes_utm = LineString(list(optimized_route_nodes_utm.geometry.values))
# people take, on average 60 minutes to cover 5 km, so it takes 12 minutes per km
# will use this to measure waking time
# distance is in meters, converting to km to 2 decimal places
shortest_route_distance = float("{:.1f}".format(shortest_route_nodes_utm.length/1000))
shortest_route_time = int(shortest_route_distance * 12)

optimized_route_distance = float("{:.1f}".format(optimized_route_nodes_utm.length/1000))
optimized_route_time = int(shortest_route_distance * 12)

shortest_route_write = ('The shortest route is ' + str(shortest_route_distance) + ' kilometres long ' +
                                'and will take approximately ' + str(shortest_route_time) + ' minutes to complete.')
optimized_route_write = ('This optimized route is ' + str(optimized_route_distance) + ' kilometres long ' +
                                'and will take approximately ' + str(optimized_route_time) + ' minutes to complete.')

# need coordinates to center map on
ave_lat = sum(start_stop["y"]) / len(start_stop["y"])
ave_lon = sum(start_stop["x"]) / len(start_stop["x"])

start_stop = start_stop.to_crs(epsg=4326)

shortest_route_nodes_projection = zip(shortest_route_nodes['y'],
                                      shortest_route_nodes['x'])

optimized_route_nodes_projection = zip(optimized_route_nodes['y'],
                                       optimized_route_nodes['x'])

########################################################
# Load map centred on average coordinates
my_map = folium.Map(location=[ave_lat, ave_lon], tiles='CartoDB positron', zoom_start=13)

# add a markers
folium.Marker([start_stop['y'][0], start_stop['x'][0]],
          icon = folium.Icon(color='green'), popup = shortest_route_write).add_to(my_map)
folium.Marker([start_stop['y'][1], start_stop['x'][1]],
          icon = folium.Icon(color='red')).add_to(my_map)
# add lines
folium.PolyLine(shortest_route_nodes_projection, color="black", weight=2.5, opacity=1).add_to(my_map)
# add lines
folium.PolyLine(optimized_route_nodes_projection, color="green", weight=3, opacity=1).add_to(my_map)

# Save map
my_map.save('notebooks/figures/demo_plot_3.html')
# Import the Folium interactive html file
from IPython.display.IFrame import HTML

HTML('<iframe src=emo_plot.html width=700 height=450></iframe>')
