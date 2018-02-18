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
from scipy.spatial import distance
from Lib import *

clusters_color = [name for name, c in dict(colors.BASE_COLORS, **colors.CSS4_COLORS).items() if "white" not in name]
random.shuffle(clusters_color)

X = 'temperature'
Y = 'pm1'
main_data = ('temperature', 'pm1')

# Nice output :)
def print_point(point):
    print("(temperature=%d, humidity=%d, pressure=%d, pm1=%d, day_of_year=%d) \t->  id=%d,\tcluster=%s" % \
        (point["temperature"], point["humidity"], point["pressure"], point["pm1"], point["day_of_year"], point["id"], str(point["cluster"])))

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

    new_point["main_data"] = [ new_point[l] for l in main_data ]

    new_point["cluster"] = None
    new_point["id"] = int(id)

    #print(new_point)
    return new_point
    
# Function generates n random points 
# and calculates density and distance to higher density point
def get_and_calculate(sc, csv_file_name):
    with open(csv_file_name) as csvfile:
        points = [ [ elem for elem in row ] for row in csv.reader(csvfile, delimiter=';') ]

    labels = points[0]
    points = [ (i, point) for i, point in enumerate(points[1:]) ]
    print("points:", int(len(points)))
    pointsRDD = sc.parallelize(points)

    copleated_points = pointsRDD.map(lambda point: compleat(point, labels))
    
    print("points compleated")
    
    plot_of_x_and_y(copleated_points.collect(), "day_of_year", "pm1", "date_and_pm1.png")
    plot_of_x_and_y(copleated_points.collect(), "day_of_year", "pm10", "date_and_pm10.png")
    plot_of_x_and_y(copleated_points.collect(), "day_of_year", "pm25", "date_and_pm25.png")
        
    return copleated_points

def choose_centers_of_clusters(points, clusters):
    min_x = int(round(points.min(key=lambda p: p[X])[X]))
    max_x = int(round(points.max(key=lambda p: p[X])[X]))
    
    min_y = int(round(points.min(key=lambda p: p[Y])[Y]))
    max_y = int(round(points.max(key=lambda p: p[Y])[Y]))

    centers = [ { X: random.randrange(min_x, max_x), Y: random.randrange(min_y, max_y), "cluster" : i+1 } for i in range(clusters) ]

    for center in centers:
        center["main_data"] = [ center[l] for l in main_data ]

    return centers

def assign_points_to_clusters(points, centers, metric=distance.euclidean):

    def set_cluster(point, centers):
        point["cluster"] = sorted(centers, key=lambda c: distance_to(c, point, metric))[0]["cluster"]

        return point
    
    def reasign_centers(points):
        for center in centers:
            points_in_cluster = points.filter(lambda point: point["cluster"] == center["cluster"] )

            try:
                center[X] = points_in_cluster.map(lambda p: p[X]).sum() / points_in_cluster.map(lambda p: p[X]).count()
                center[Y] = points_in_cluster.map(lambda p: p[Y]).sum() / points_in_cluster.map(lambda p: p[Y]).count()
            
                center["main_data"] = [ center[l] for l in main_data ]
            except: continue

    any_change = False
    
    was = { point["id"] : point["cluster"] for point in points.toLocalIterator() }
    while not any_change:
        points = points.map(lambda point: set_cluster(point, centers))

        now = { point["id"] : point["cluster"] for point in points.toLocalIterator() }

        for key in now.keys():
            if was[key] != now[key]:
                print("There was a change!")
                break
        else:
            print("There was no change!")
            return points

        reasign_centers(points)

        was = now

def main(sc, csv_file_name, clusters, metric=distance.euclidean):
    func_name = str(metric).split(" ")[1]
    points = get_and_calculate(sc, csv_file_name)

    make_basic_plots(points)

    centers = choose_centers_of_clusters(points, clusters)
    print("centers done", centers)

    print("Assign with", func_name, "metric")
    points = assign_points_to_clusters(points, centers, metric)
    print("clusters done")
    plot_clusters(points, X, Y, 'clusters_K-mean_' + func_name + '.png')  

    points = points.collect()
    print("\n\nPoints:")
    i = 1
    for cluster in range(1, clusters+1) + [None]:
        for point in points:
            if point["cluster"] == cluster:
                print("%4d)" % i, end=' ')
                print_point(point)
                i += 1
        print("\n\n")

if __name__ == "__main__":
    conf = SparkConf().setAppName('DataMining_Project')
    sc = SparkContext(conf=conf)

    for metric in [distance.euclidean, distance.cityblock, distance.cosine, distance.chebyshev]:
        main(sc, csv_file_name='full-year-2017-studencka-189-small.csv', clusters=4, metric=metric)

    matplotlib.pyplot.close('all')