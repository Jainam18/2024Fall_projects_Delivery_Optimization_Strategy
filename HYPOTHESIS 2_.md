HYPOTHESIS 2

"The total time taken for completing deliveries is less using the travelling salesman problem route optimization approach than greedy route strategy but the as the number of deliveries volume increases the travelling sales problem approach takes a lot of computation time"

Greedy approach - Out of X number of delivery points, we will consider the point nearest to the hub as first delivery location. Then after delivering on first location, out of X-1 delivery points, we will consider the next nearest point from the current location of vehicle( which would be 1st delivery points)

Travelling sales man approach- Given a list of delivery locations and the distances between each pair of locations, what is the shortest possible route that visits each location exactly once and returns to the hub.

Random variables-
1. Traffic Congestion
2. Delivery locations
3. Speed variability 


Output-
1. Time taken for a particular location - total distance to reach a delivery location from last source(source can be hub if it is 1st location or it could be delivery location that was just delivered an order)/ speed for that particular route

2. Total time taken to deliver all orders- summation of time taken for all locations 

Results-






references-
Travelling sales man problem-
https://en.wikipedia.org/wiki/Travelling_salesman_problem#:~:text=In%20the%20theory%20of%20computational,returns%20to%20the%20origin%20city%3F%22 