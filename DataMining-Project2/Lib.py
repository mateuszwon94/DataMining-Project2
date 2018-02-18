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

clusters_color = [name for name, c in dict(colors.BASE_COLORS, **colors.CSS4_COLORS).items() if "white" not in name]
random.shuffle(clusters_color)

# Dinstance between two points
def distance_to(point_i, point_j, metric=distance.euclidean):
    return metric(point_i["main_data"], point_j["main_data"])

def get_color(point, clusters_color):
    if point["cluster"] is None:
        return clusters_color[0]
    return clusters_color[point["cluster"]]

def plot_clusters(points, x_label, y_label, file_name):
    x = [point[x_label] for point in points.collect()]
    y = [point[y_label] for point in points.collect()]
    c = [get_color(point, clusters_color) for point in points.collect()]
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(x, y, color=c, s=5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(file_name)
    print("Image '%s' saved!" % file_name)

# Simple plot of generated points
def plot_of_x_and_y(points, x_label, y_label, file_name):
    x = [point[x_label] for point in points]
    y = [point[y_label] for point in points]
    c = ['black' if point['cluster'] is None else get_color(point, clusters_color) for point in points]
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(x, y, color=c)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(file_name)
    print("Image '%s' saved!" % file_name)

def plot_temperature_humidity_pressure(points, file_name):
    points = points.collect()
    x = [point["day_of_year"] for point in points]
    y1 = [point["temperature"] for point in points]
    y2 = [point["humidity"] for point in points]
    y3 = [point["pressure"] for point in points]
    plt.figure(1)
    plt.subplot(311)
    plt.plot(x, y1, 'r')
    plt.xlabel('day_of_year')
    plt.ylabel('temperature')
    plt.subplot(312)
    plt.plot(x, y2, 'b')
    plt.xlabel('day_of_year')
    plt.ylabel('humidity')
    plt.subplot(313)
    plt.plot(x, y3, 'g')
    plt.xlabel('day_of_year')
    plt.ylabel('pressure')
    plt.savefig(file_name)
    print("Image '%s' saved!" % file_name)