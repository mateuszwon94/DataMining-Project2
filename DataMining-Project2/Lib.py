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

clusters_color = [name for name, c in dict(colors.BASE_COLORS, **colors.CSS4_COLORS).items() if "white" not in name]
random.shuffle(clusters_color)

# Dinstance between two points
def distance_to(point_i, point_j, x, y):
    return math.sqrt((point_i[x] - point_j[x])**2 + (point_i[y] - point_j[y])**2)

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
    c = [get_color(point, clusters_color) for point in points]
    fig, ax = matplotlib.pyplot.subplots()
    ax.scatter(x, y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    fig.savefig(file_name)
    print("Image '%s' saved!" % file_name)