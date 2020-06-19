import pandas
import osmnx
import pickle
import numpy
import geopandas
from photutils.utils import ShepardIDWInterpolator
import gdal # dealing with raster data
import rasterstats
import networkx
##################################################################################################
##################################################################################################
                            # function to create geopandas object #

def create_gdf(df, Longitude, Latitude, projection):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[Longitude], df[Latitude]),
                                  crs=projection)

##############################################################################################
robbery_model_nodes_2958 = pandas.read_pickle("C:/Users/jodyn/Google Drive/Insight Data Science/Insight/Processed Data/robbery_model_nodes_2958.pkl")
pedestrian_model_nodes_2958 = pandas.read_pickle("C:/Users/jodyn/Google Drive/Insight Data Science/Insight/Processed Data/pedestrian_model_nodes_2958.pkl")
##################################################################################################
##################################################################################################
place_name = 'City of Toronto'
walk_path_GTA = osmnx.graph_from_place(place_name, network_type='walk')
# need to reproject to UTM for later analysis
walk_path_GTA_proj = osmnx.project_graph(walk_path_GTA)
# need edges and nodes to append data to
walk_nodes_proj, walk_edges_proj = osmnx.graph_to_gdfs(walk_path_GTA, nodes=True, edges=True)

# making sure its in NAD 1983 and save
walk_nodes_proj = walk_nodes_proj.to_crs(epsg = 2958)

#walk_nodes_proj['Index'] = walk_nodes_proj.index

# converting the node data to a geopanda dataframe so that I can extract elevation data
walk_node_id = pandas.DataFrame(walk_nodes_proj)
walk_nodes_gdf = create_gdf(df=walk_node_id,
                            Latitude="y",
                            Longitude="x",
                            projection="EPSG:4326")
walk_nodes_gdf = walk_nodes_gdf.to_crs(epsg=2958)

###################################################################################################
# I will use IDW to interpolate the probabilty values at each node.
# I will then extract each of the terrain values at each node by simply using a raster extract.
# # first I need to set up the IDW with known values
# first I need to set up the IDW with known values
coords_mugg = numpy.column_stack((robbery_model_nodes_2958['geometry'].x,robbery_model_nodes_2958['geometry'].y))
coords_collison = numpy.column_stack((pedestrian_model_nodes_2958['geometry'].x,pedestrian_model_nodes_2958['geometry'].y))
coords_walk_nodes = numpy.column_stack((walk_nodes_gdf['geometry'].x,walk_nodes_gdf['geometry'].y))

values_mugg = list(robbery_model_nodes_2958['Mugging'])
values_collison = list(pedestrian_model_nodes_2958['Collison'])

idw_mugging = ShepardIDWInterpolator(coords_mugg,values_mugg)
idw_collision = ShepardIDWInterpolator(coords_collison,values_collison)

interpolated_mugging = idw_mugging(coords_walk_nodes,n_neighbors=19,power=1.0,reg=5.,eps=0.1)
interpolated_collision = idw_collision(coords_walk_nodes,n_neighbors=19,power=1.0,reg=5.,eps=0.1)

##########################################################################################################
###################################################################################################
# Nodes & Edges Added For Terrain Data #
###################################################################################################
# add elevation to each of the nodes, using the google elevation API, then calculate edge grades
# will extract values of hillshade and TRI at each node in GIS
filepath_Roughness = r"C:/Users/jodyn/Google Drive/Insight Data Science/Insight/Terrain/DTM_Roughness_2958.tif"
filepath_Hillshade = r"C:/Users/jodyn/Google Drive/Insight Data Science/Insight/Terrain/DTM_Hillshade_2958.tif"
raster_Roughness = gdal.Open(filepath_Roughness)
raster_Hillshade = gdal.Open(filepath_Hillshade)

walk_nodes_gdf['Roughness'] = rasterstats.point_query(walk_nodes_gdf, filepath_Roughness,
                                                interpolate='nearest')
walk_nodes_gdf['Hillshade'] = rasterstats.point_query(walk_nodes_gdf, filepath_Hillshade,
                                                      interpolate='nearest')

walk_nodes_gdf['Roughness'] = walk_nodes_gdf['Roughness'].fillna(0)
walk_nodes_gdf['Hillshade'] = walk_nodes_gdf['Hillshade'].fillna(0)

################################################################################################
################################################################################################
walk_nodes_gdf['Collison'] = interpolated_collision.astype(float)
walk_nodes_gdf['Mugging'] = interpolated_mugging.astype(float)
################################################################################################
################################################################################################
# find the edge weights based on these model values
walk_edges_proj['Rob'] = [int(100 * (walk_nodes_gdf.loc[u]['Mugging'] + walk_nodes_gdf.loc[v]['Mugging']))
                       for u, v in zip(walk_edges_proj['u'], walk_edges_proj['v'])]
walk_edges_proj['Collison'] = [int(100 * (walk_nodes_gdf.loc[u]['Collison'] + walk_nodes_gdf.loc[v]['Collison']))
                       for u, v in zip(walk_edges_proj['u'], walk_edges_proj['v'])]
walk_edges_proj['Hillshade'] = [int(0.3937 * (walk_nodes_gdf.loc[u]['Hillshade'] + walk_nodes_gdf.loc[v]['Hillshade']))
                       for u, v in zip(walk_edges_proj['u'], walk_edges_proj['v'])]
walk_edges_proj['Roughness'] = [int(10 * (walk_nodes_gdf.loc[u]['Roughness'] + walk_nodes_gdf.loc[v]['Roughness'])) # TSC ranges were super larg
                       for u, v in zip(walk_edges_proj['u'], walk_edges_proj['v'])]
#####################################################################################################
#####################################################################################################

Robs = {}
Collisons = {}
Hillshades = {}
Roughnesss = {}

# Set each edge's tree weight as the average of the tree weights of the edge's vertices
for row in walk_edges_proj.itertuples():
    u = getattr(row, 'u')
    v = getattr(row, 'v')
    key = getattr(row, 'key')
    Rob = getattr(row, 'Rob')
    Collison = getattr(row, 'Collison')
    Hillshade = getattr(row, 'Hillshade')
    Roughness = getattr(row, 'Roughness')

    Robs[(u, v, key)] = Rob
    Collisons[(u, v, key)] = Collison
    Hillshades[(u, v, key)] = Hillshade
    Roughnesss[(u, v, key)] = Roughness

# need to set the edge attributes
networkx.set_edge_attributes(walk_path_GTA_proj, Robs, 'Rob')
networkx.set_edge_attributes(walk_path_GTA_proj, Collisons, 'Collison')
networkx.set_edge_attributes(walk_path_GTA_proj, Hillshades, 'Hillshade')
networkx.set_edge_attributes(walk_path_GTA_proj, Roughnesss, 'Roughness')

##################################################################################################
walk_edges_proj.to_pickle("webapplication/flaskexample/data/edges.pkl")
walk_nodes_proj.to_pickle("webapplication/flaskexample/data/nodes.pkl")
with open("webapplication/flaskexample/data/path.p", 'wb') as f:
    pickle.dump(walk_path_GTA_proj,f)

######################################################################################################

orig_xy = osmnx.utils_geo.geocode("CN Tower, Toronto, Canada")
target_xy = osmnx.utils_geo.geocode("Casa Loma, Toronto, Canada")

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

G_edges = walk_edges_proj
G_nodes = walk_nodes_proj
G_walk = walk_path_GTA_proj

# Find the node in the graph that is closest to the origin point (here, we want to get the node id)
orig_node = osmnx.get_nearest_node(G_walk, orig_xy, method='euclidean')
# Find the node in the graph that is closest to the target point (here, we want to get the node id)
target_node = osmnx.get_nearest_node(G_walk, target_xy, method='euclidean')
##################################################################################################
# Calculate new edge weights
Collisons = {}
Hillshades = {}
Robs = {}
Roughnesss = {}
lengths = {}
for row in G_edges.itertuples():
    u = getattr(row, 'u')
    v = getattr(row, 'v')
    key = getattr(row, 'key')
    Collison = getattr(row, 'Collison')
    Hillshade = getattr(row, 'Hillshade')
    Rob = getattr(row, 'Rob')
    Roughness = getattr(row, 'Roughness')
    length = getattr(row, 'length')

    Collisons[(u, v, key)] = Collison
    Hillshades[(u, v, key)] = Hillshade
    Robs[(u, v, key)] = Rob
    Roughnesss[(u, v, key)] = Roughness
    lengths[(u, v, key)] = length

# Optimized attribute is a weighted combo of normal length, tree counts, and road safety.
# Larger value is worse

optimized = {}
for key in lengths.keys():
    temp = (lengths[key]) # 2450
    temp += (6 * (Robs[key])) # 186
    temp += (24 * (Roughnesss[key])) # 101
    temp += (13 * (Collisons[key])) # 196
    temp += (13 * (Hillshades[key]))# 199
    optimized[key] = temp


