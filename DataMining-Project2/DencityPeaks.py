#!/usr/bin/env python2
# coding UTF-8

from __future__ import division, print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy import genfromtxt
import math
import random
import time
import csv
import datetime
from sys import float_info
from pyspark import SparkContext, SparkConf
from Lib import *

X = 'temperature'
Y = 'pm1'
main_data = ('temperature', 'pm1')

# Nice output :)
def print_point(point, x, y):
    print("(%d, %d)  -> id=%d,  \tid_of_closest_neighbor=%s,\tcluster=%s,\tdensity=%d,\tdistance_to_higher_density_point=%.4f" % \
        (point[x], point[y], point["id"], str(point["id_of_point_with_higher_density"]),
         str(point["cluster"]), point["density"], point["distance_to_higher_density_point"]))

# Function computing density for points
# Density is computed as a number of points which is closed to current than cutoff_distance
def set_density(point_i, points, cutoff_distance):
    point_i["density"] = 0

    for point_j in points:
        if point_i == point_j: continue

        if distance_to(point_i, point_j) < cutoff_distance: point_i["density"] += 1

    return point_i

# Function computing distance_to_higher_density_point for points
# distance_to_higher_density_point is computed as minimum distance to point with higher density
# If point has the highest density this is computed as max distance to any other point
def set_distance_to_higher_density_point(point_i, points):
    point_i["distance_to_higher_density_point"] = float("inf")
        
    for point_j in points:
        if point_i == point_j: continue

        if point_i["density"] >= point_j["density"]: continue

        distance = distance_to(point_i, point_j)
        if distance < point_i["distance_to_higher_density_point"]:
            point_i["distance_to_higher_density_point"] = distance
            point_i["id_of_point_with_higher_density"] = int(point_j["id"])

    if point_i["distance_to_higher_density_point"] == float("inf"):
       point_i["distance_to_higher_density_point"] = 0

    for point_j in points:
        if point_i == point_j: continue
            
        if distance_to(point_i, point_j) > point_i["distance_to_higher_density_point"]:
            point_i["distance_to_higher_density_point"] = distance_to(point_i, point_j)

    return point_i

def compleat(point, labels):
    (id, point) = point
    new_point = { labels[i] : elem for i, elem in enumerate(point) }

    for key in new_point.keys():
        try:
            new_point[key] = float(new_point[key])
            if "pressure" in key:
                new_point[key] /= 100.
        except: pass
        
    (D, T) = new_point["time"].split("T")
    (year, month, day) = D.split("-")
    (hours, minutes, seconds) = T.split(":")
    new_point["time"] = datetime.datetime(int(year), int(month), int(day), int(hours), int(minutes), int(float(seconds)))
    new_point["date"] = new_point["time"].date()
    new_point["time"] = new_point["time"].time()
    new_point["day_of_year"] = int(new_point["date"].strftime('%j'))

    new_point["pm1"] = int(new_point["pm1"])
    new_point["pm10"] = int(new_point["pm10"])
    new_point["pm25"] = int(new_point["pm25"])

    new_point["density"] = None
    new_point["distance_to_higher_density_point"] = None
    new_point["id_of_point_with_higher_density"] = None

    new_point["main_data"] = [ new_point[l] for l in main_data ]

    new_point["cluster"] = None
    new_point["id"] = int(id)

    #print(new_point)
    return new_point

# Function generates n random points 
# and calculates density and distance to higher density point
def get_and_calculate(sc, csv_file_name, cutoff_distance):
    with open(csv_file_name) as csvfile:
        points = [ [ elem for elem in row ] for row in csv.reader(csvfile, delimiter=';') ]

    labels = points[0]
    points = [ (i, point) for i, point in enumerate(points[1:]) ]
    print("points:", int(len(points)))
    pointsRDD = sc.parallelize(points)

    copleated_points = pointsRDD.map(lambda point: compleat(point, labels))
    list_of_copleated_points = [point for point in copleated_points.toLocalIterator()]
    
    print("points compleated")
    
    plot_of_x_and_y(list_of_copleated_points, "day_of_year", "pm1", "date_and_pm1.png")
    plot_of_x_and_y(list_of_copleated_points, "day_of_year", "pm10", "date_and_pm10.png")
    plot_of_x_and_y(list_of_copleated_points, "day_of_year", "pm25", "date_and_pm25.png")

    points_with_local_density = copleated_points.map(
        lambda point: set_density(point, list_of_copleated_points, cutoff_distance))
    list_of_random_points = [point for point in points_with_local_density.toLocalIterator()]
    
    print("local density computed")

    points_with_distance_to_higher_density_point = points_with_local_density.map(
        lambda point: set_distance_to_higher_density_point(point, list_of_random_points))

    print("points with higher density set")
    
    return points_with_distance_to_higher_density_point

def plot_of_density_and_distance_to_higher_density_point(points, file_name):
    points = points.collect()
    x = [point["density"] for point in points]
    y = [point["distance_to_higher_density_point"] for point in points]
    c = ['green' if point["cluster"] is not None else 'red' for point in points]
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(x, y, c=c, s=5)
    ax.set_xlabel('density')
    ax.set_ylabel('distance_to_higher_density_point')
    fig.savefig(file_name)
    print("Image '%s' saved!" % file_name)

def choose_centers_of_clusters(points, n):
    
    # TODO: try to write function, which automatically chooses number of centers
    def set_center(point, centers):
        for i, center in enumerate(centers):
            if point["id"] == center["id"]:
                    point["cluster"] = i+1

        return point
    
    sorted_points = points.sortBy(
        lambda p: -(p["density"] * p["distance_to_higher_density_point"]))
    centers = sorted_points.take(n)

    points = points.map(
        lambda point: set_center(point, centers))

    print("\n\nCenters of clusters:")
    for point in points.collect():
        if point["cluster"]:
            print_point(point, X, Y)

    return points

def assign_points_to_clusters(points):
        
    def set_cluster(point, all_points):
        if point["cluster"] is not None or point["id_of_point_with_higher_density"] is None:
            return point

        ref_point = [ p for p in all_points if p["id"] == point["id_of_point_with_higher_density"] ][0]
        point["cluster"] = ref_point["cluster"]

        return point
    
    was = points.filter(lambda point: point["cluster"] == None).count()
    print(was)

    while None in [point["cluster"] for point in points.collect()]:
        all_points = points.collect()
        points = points.map(lambda point: set_cluster(point, all_points))

        now = points.filter(lambda point: point["cluster"] == None).count()
        print(now)
        if was == now: break
        was = now

    return points

def complexity_in_time(sc, clusters, limit, cutoff_distance, p_min, p_max, k):
    complexity_data = {"points": [], "time": []}
    
    for i in xrange(p_min, p_max + 1, k):
        try:
            start_time = time.time()
            points = generate_and_calculate(sc, i, limit, cutoff_distance)
            points = choose_centers_of_clusters(points, clusters)
            points = assign_points_to_clusters(points)
        
            elapsed_time = time.time() - start_time
            complexity_data["points"].append(i)
            complexity_data["time"].append(elapsed_time)

            plot_clusters(points, 'clusters_T-%d.png' % i)
            print("Points: %d  --->  time = %f"% (i, elapsed_time))
            matplotlib.pyplot.close('all')
           
        except Exception:
            print("Exception!!!")
            continue

    avg = sum(complexity_data["time"])/len(complexity_data["time"])
    print("Average time is = %f"% (i, avg))

    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(complexity_data["points"], complexity_data["time"], s=5)
    ax.set_xlabel('points')
    ax.set_ylabel('time')
    fig.savefig('complexity_time-%d.png' % clusters)

def complexity_in_clusters_number(sc, clusters_min, clusters_max, limit, cutoff_distance, p, k=1):
    complexity_data = {"points": [], "time": []}
        
    for i in xrange(clusters_min, clusters_max + 1, k):
        try:
            start_time = time.time()
            points = generate_and_calculate(sc, p, limit, cutoff_distance)
            points = choose_centers_of_clusters(points, i)
            points = assign_points_to_clusters(points)
        
            elapsed_time = time.time() - start_time
            complexity_data["points"].append(i)
            complexity_data["time"].append(elapsed_time)

            plot_clusters(points, 'clusters_C-%d.png' % i)
            print("Clusters: %d  --->  time = %f"% (i, elapsed_time))
            matplotlib.pyplot.close('all')
           
        except Exception:
            print("Exception!!!")
            continue

    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(complexity_data["points"], complexity_data["time"], s=5)
    ax.set_xlabel('clusters')
    ax.set_ylabel('time')
    fig.savefig('complexity_clusters-%d.png' % p)

def check_of_cutoff_distance(sc, clusters, limit, cutoff_distance_min, cutoff_distance_max, p, k = 1):
    list_of_random_points = generate_list_of_random_points(p, limit)
    pointsRDD = sc.parallelize(list_of_random_points)
    
    for i in xrange(cutoff_distance_min, cutoff_distance_max + 1, k):
        #try:
            points_with_local_density = pointsRDD.map(
                lambda point: set_density(point, list_of_random_points, i))
            list_of_random_points = [point for point in points_with_local_density.toLocalIterator()]

            points = points_with_local_density.map(
                lambda point: set_distance_to_higher_density_point(point, list_of_random_points))
            points = choose_centers_of_clusters(points, clusters)
            points = assign_points_to_clusters(points)
        
            plot_clusters(points, 'clusters_C-%f.png' % i)
            print("Cut Off: %f"% (i))
            matplotlib.pyplot.close('all')
           
        #except Exception:
        #    print("Exception!!!")
        #    continue

def main(sc, csv_file_name, clusters, cutoff_distance):
    points = get_and_calculate(sc, csv_file_name, cutoff_distance)
    points = choose_centers_of_clusters(points, clusters)
    print("centers done")
    plot_of_density_and_distance_to_higher_density_point(points, 'density.png')

    points = assign_points_to_clusters(points)
    print("clusters done")
    plot_clusters(points, X, Y, 'clusters-dencity_peak.png')  
    plot_of_x_and_y(points, "day_of_year", "pm1", "date_and_pm1_clusters.png")

    print("\n\nPoints:")
    for point in points.collect():
        print_point(point, X, Y)

if __name__ == "__main__":
    conf = SparkConf().setAppName('DataMining_Project')
    sc = SparkContext(conf=conf)

    main(sc, csv_file_name='full-year-2017-studencka-189-small.csv', clusters=4, cutoff_distance=10)

    #complexity_in_time(sc, clusters=5, limit=10, cutoff_distance=1, p_min=20, p_max=1000, k=20)
    #complexity_in_clusters_number(sc, clusters_min=3, clusters_max=50, limit=10, cutoff_distance=1, p=300)
    #check_of_cutoff_distance(sc, clusters=5, limit=10, cutoff_distance_min=1, cutoff_distance_max=10, p=300)

    matplotlib.pyplot.close('all')
