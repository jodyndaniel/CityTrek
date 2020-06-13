# Insight Data Science Fellowship

This repository is for the web application **CityTrek**.
 Imagine you had an unexpected layover in downtown Toronto,
 you would like to **walk**  from the Casa Loma to the CN tower,
 but you have no idea whether the path your navigation 
 application has recommended is comfortable or safe. 
 Want a better option?
 **CityTrek** is the application for you! 
 
 **CityTrek** takes your comfort 
 (e.g., preferences for shady or flatter paths)
 and safety (e.g., risk of mugging and car collision) preferences
 and recommends an optimized path compared to the shortest path.
 **CityTrek** will also provide details on 
 how this optimized path compares to the shortest path, in terms
 of distance and walking time. 
 
 Because we understand that crime data (e.g., mugging) is highly
 biased, which could result in paths that avoid marginalized 
 neighbourhoods, **CityTrek**  places a much 
 lower weight on this feature compared to the others. Simply,
 marginalized communities are often over-policed, which 
 leads to [higher incidences of reported of crimes](https://www.jstor.org/stable/41954178?seq=1).
This bias in the data would result in the model predicting that
such neighbourhoods are unsafe, when in fact, they are simply
over-policed.
 
 The other weights (i.e., distance, hilliness, risk of pedestrian car collision, shadiness)
 are given equal value in the application. Importantly, the metric used to measure shadiness
 does not take into consideration time of day. Check out 
 [Parasol](https://blog.insightdatascience.com/parasol-navigation-optimizing-walking-routes-to-keep-you-in-the-sun-or-shade-1be7a4fde97), 
 which is an application made by a past Insight Data Science Fellow that plots your path based on 
 shade preferences - it takes into consideration the time of day.
