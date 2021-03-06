from flask import render_template, request
import osmnx
import geopandas
import pandas
import networkx
import folium
import pickle
from shapely.geometry import LineString
from flaskexample import app

# weighted road network for shortest path
G_nodes = pandas.read_pickle(app.root_path + '/' + 'data/nodes.pkl')
G_edges = pandas.read_pickle(app.root_path + '/' + 'data/edges.pkl')
with open(app.root_path + '/' + 'data/path.p', 'rb') as f:
    G_walk = pickle.load(f)

# need to call the edge weights to that I can apply the optimized weights
# for the later shortest path measures in the applicaton
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

# function to create geopandas dataframe
def create_gdf(df, Longitude, Latitude, projection):
    return geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df[Longitude], df[Latitude]),
                                  crs=projection)

@app.route('/', methods=["GET", "POST"])  # we are now using these methods to get user input
def home_page():
    return render_template('index.html')  # render a template

@app.route('/output', methods=["POST"])
def recommendation_output():
    global length, Roughness, Rob, key, v, u
    # Pull input
    car = int(request.form.get('car'))
    mug = int(request.form.get('mug'))
    shade = int(request.form.get('shade'))
    rough = int(request.form.get('rough'))
    dist = int(request.form.get('dist'))

    orig_xy_form = request.form.get('start_loc')
    target_xy_form = request.form.get('end_loc')
    orig_xy_geo = "%s, Toronto, Canada" % orig_xy_form
    target_xy_geo = "%s, Toronto, Canada" % target_xy_form
    try:
        orig_xy = osmnx.geocode(orig_xy_geo)
    except:
        orig_xy = " "
    try:
        target_xy = osmnx.geocode(target_xy_geo)
    except:
        target_xy = " "

    if (orig_xy == target_xy):
        start_coords = (43.7112075, -79.4762563)
        folium_map = folium.Map(location=start_coords, tiles='CartoDB positron', zoom_start=14)
        map_path = app.root_path + '/' + 'static/map_demo_0.html'
        folium_map.save(map_path)
        text_output = ('It seems that you entered the same address as your start and end location.'
                       ' Consequently, we have returned an empty map! Try again?')
        return render_template('index.html',
                               my_output='map_demo_0.html',
                               my_text_null= text_output,
                               my_form_result="Empty")
    elif (orig_xy==" " or target_xy ==" "):
        start_coords = (43.7112075, -79.4762563)
        folium_map = folium.Map(location=start_coords, tiles='CartoDB positron', zoom_start=14)
        map_path = app.root_path + '/' + 'static/map_demo_1.html'
        folium_map.save(map_path)
        text_output = ('One or both of locations you entered were not found in the City of Toronto. Consequently, we '
                       'returned an empty map! Try again?')
        return render_template('index.html',
                               my_output='map_demo_1.html',
                               my_text_null=text_output,
                               my_form_result="Empty")
    elif (orig_xy_geo==(43.6534817, -79.3839347) or target_xy==(43.6534817, -79.3839347)):
        start_coords = (43.7112075, -79.4762563)
        folium_map = folium.Map(location=start_coords, tiles='CartoDB positron', zoom_start=14)
        map_path = app.root_path + '/' + 'static/map_demo_2.html'
        folium_map.save(map_path)
        text_output = ('You did not enter a start or end location.'
                       ' Consequently, we returned an empty map! Try again?')
        return render_template('index.html',
                               my_output='map_demo_2.html',
                               my_text_null=text_output,
                               my_form_result="Empty")
    # all weights below are aimed at scaling each of the features. Risk of mugging has a lower weight because
    # these data are biased (weighted as half less important)
    else:
        # need location UTMs to find nodes
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

        # Optimized attribute is a weighted combo of path length, risk of being mugged/hit by a car, shadiness and hilliness.
        # Larger value is worse
        optimized = {}
        for key in lengths.keys():
            temp = int(dist * lengths[key])
            temp += int(mug * 6 * (Robs[key]))
            temp += int(car * 13 * (Collisons[key]))
            temp += int(rough * 24 * (Roughnesss[key]))
            temp += int(shade * 13 * Hillshades[key])
            optimized[key] = temp
        networkx.set_edge_attributes(G_walk, optimized, 'optimized')
        # Path of nodes
        optimized_route = networkx.shortest_path(G_walk, orig_node, target_node, weight='optimized')
        # Path of nodes
        shortest_route = networkx.shortest_path(G_walk, orig_node, target_node, weight='length')

        # Get the nodes along the routes path
        shortest_route_nodes = G_nodes.loc[shortest_route]
        optimized_route_nodes = G_nodes.loc[optimized_route]
        # need coordinates to center map on
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
        shortest_route_distance = float("{:.1f}".format(shortest_route_nodes_utm.length / 1000))
        shortest_route_time = int(shortest_route_distance * 12)

        optimized_route_distance = float("{:.1f}".format(optimized_route_nodes_utm.length / 1000))
        optimized_route_time = int(optimized_route_distance * 12)

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
                      icon=folium.Icon(color='blue')).add_to(my_map)
        folium.Marker([start_stop['y'][1], start_stop['x'][1]],
                      icon=folium.Icon(color='blue')).add_to(my_map)
        # add lines
        folium.PolyLine(shortest_route_nodes_projection, color="grey", weight=2.5, opacity=1).add_to(my_map)
        # add lines
        folium.PolyLine(optimized_route_nodes_projection, color="green", weight=3, opacity=1).add_to(my_map)
        # Save map
        map_path = app.root_path + '/' + 'static/map_demo_3.html'
        my_map.save(map_path)
        return render_template('index.html',
                               my_output='map_demo_3.html',
                               my_textoutput_shortest = shortest_route_write,
                               my_textoutput_optimized = optimized_route_write,
                               my_form_result="NotEmpty")

