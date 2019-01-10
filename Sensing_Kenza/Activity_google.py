##In this code, the data is extracted from the json file present in this folder, they are split by days, analysed to find out the length of activity

##Step 1 - import all the necessary modules to carry out the different tasks

import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import datetime
import math
import matplotlib.pyplot as plt
import operator
from collections import Counter
import requests
import urllib.parse
import pandas as pd
from pymongo import MongoClient
from pprint import pprint



##Step 2 - initialise all the variables
latitudes=[]
longitudes=[]
velocities =[]
altitudes=[]
stamp_ms=[]
stamps=[]
time_ms=[]

##Step 3 - import the json files containing the data activity collected by googlempas

activity_first_part = json.loads(open('Locations_Chamonix_velocity.json').read())
activity_second_part = json.loads(open('activity_clarisse.json').read())

##Step 4 - append each type of data to a separate list to be easily used
for x in range(0, len(activity_first_part)):
    latitudes.append((activity_first_part[x]['latitudeE7'])*0.0000001)
    longitudes.append((activity_first_part[x]['longitudeE7'])*0.0000001)
    stamp_ms.append(activity_first_part[x]['timestampMs'])
    velocities.append(activity_first_part[x]['velocity'])
    altitudes.append(activity_first_part[x]['altitude'])

for x in range(0, len(activity_second_part)):    
    latitudes.append((activity_second_part[x]['latitudeE7'])*0.0000001)
    longitudes.append((activity_second_part[x]['longitudeE7'])*0.0000001)
    stamp_ms.append(activity_second_part[x]['timestampMs'])
    velocities.append(activity_second_part[x]['velocity'])
    altitudes.append(activity_second_part[x]['altitude'])

    
##Step5 - extract the relevant informations from the lists
for x in range(0, len(stamp_ms)):
    stamps.append(str(stamp_ms[x])[:10])

#Translate the ms timestamps into dates and hours of the day

time_ms2=[]
for x in range(0, len(stamps)):
    readable = datetime.datetime.fromtimestamp(int(stamps[x])).isoformat()
    readable2 = datetime.datetime.strptime(readable, "%Y-%m-%dT%H:%M:%S")
    #readable2 =  readable.replace('T',' ')
    #print(readable2)
    time_ms.append(readable)
    time_ms2.append(readable2)

longitude_day = [['day1'],['day2'],['day3'],['day4'],['day5'],['day6'],['day7'],['day8'],['day9'],['day10'],['day11']]
days = ['16','17','18','19','20','21','22','23','28','29','30']
date_ms=[]
for x in range(0,len(time_ms)):
    date_ms.append(str(time_ms[x])[8:10])

coordinates=[]
coordinate_1=[]

for s in range(0, len(longitudes)):
    coordi = [latitudes[s],longitudes[s]]
    #print (coordi)
    coordinate_1.append(coordi)
    
    coordinate= str([latitudes[s],longitudes[s]])
    coordinates_bis = coordinate.replace('[','')
    coordinates_biss = coordinates_bis.replace(']','')
    coordinates.append(coordinates_biss)

coordinates_day = [['day1'],['day2'],['day3'],['day4'],['day5'],['day6'],['day7'],['day8'],['day9'],['day10'],['day11']]

for i in range(0,len(date_ms)):
    for j in range(0,11):
        if date_ms[i] == days[j]:
            coordinates_day[j].append(coordinate_1[i])



latitude_day = [['day1'],['day2'],['day3'],['day4'],['day5'],['day6'],['day7'],['day8'],['day9'],['day10'],['day11']]

for i in range(0,len(date_ms)):
    for j in range(0,11):
        if date_ms[i] == days[j]:
            longitude_day[j].append(latitudes[i])


addresse=[]          

for d in range(0,len(date_ms)):
    params = {
        'latlng': coordinates[d],
        'key' : 'AIzaSyC7r7iGjazMjzaAwK4DnkXv8Ubv32f81UM',
        }
    query = urllib.parse.urlencode(params)
    url_query = urllib.parse.unquote(query)

    GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json?' + url_query
    #print(GOOGLE_MAPS_API_URL)
    req = requests.get(GOOGLE_MAPS_API_URL)
    res= req.json()

    #print(res)
    result = res['results'][0]

    addresses = result['formatted_address']
    #print(addresses)
    #print (addresses)

    addresse.append(addresses)

##Step9 - Apply  nearest neighbors algorithm

#Define the relevant functions
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors

coordinate_update=[]
coordinate_update = coordinates_day[1:6]

addresse_update=[]
addresse_update = addresse[1:6]

#print(addresse_update)

for i in range(0,len(coordinate_update)):
    for j in range(0,11):
        if date_ms[i] == days[j]:
            longitude_day[j].append(latitudes[i])

addresse_time = {}
time_stamp = {}
coordi_addresse={}
coordi_time={}
time_velocity={}
time_velocity2={}

for i in range(0,len(time_ms)):
    addresse_time[addresse[i]] = [time_ms[i]]
    time_stamp[time_ms[i]]=stamps[i]
    coordi_addresse[coordinates[i]]=addresse[i]
    coordi_time[coordinates[i]]=time_ms[i]
    time_velocity[time_ms[i]]=velocities[i]
    time_velocity2[time_ms2[i]]=velocities[i]




#apply the functions on the relevant lists
    

nbofitems_2=[15,10,2,35,7]

testInstances_2 =[[45.961214399999996, 6.8869466],[45.803365799999995, 6.9349422],[45.941119199999996, 6.854474499999999],[45.849677199999995, 6.614935699999999],[45.9665372, 6.9433641999999995]]
timess=[]
durations=[]


for p in range(0,5):
    itemk = (coordinate_update[p][1:len(coordinate_update[p])])
    similar = testInstances_2[p]
    #print(similar)
    ghj = nbofitems_2[p]
    neighbors = getNeighbors(itemk, similar, ghj)
    #print (neighbors)
    neighbors_2=[]
    for t in neighbors:
        pol = str(t)
        pol2 = pol.replace('[','')
        pol3 = pol2.replace(']','')
        neighbors_2.append(pol3)
    #print (neighbors_2)
    same_activity=[]
    for f in range (0, len(neighbors_2)):
        lol = coordi_time[neighbors_2[f]]
        same_activity.append(lol)
    #print (same_activity)
    same1=str(min(same_activity))
    same_1 = same1.replace('[','')
    same_11 = same_1.replace(']','')

    same2=str(max(same_activity))
    same_2 = same2.replace('[','')
    same_22 = same_2.replace(']','')

    #print(same_11)
    #print (same_22)
    timess.append(same_11)
    timess.append(same_22)
    same4 = time_stamp.get(same_11)
    same3 = time_stamp.get(same_22)

    #print(same4)
    #print(same3)

    same5 = int(same3) - int(same4)
    same6 = datetime.datetime.fromtimestamp(same5).isoformat()
    print(str((same6)[11:19]))
    durations.append(str((same6)[11:19]))




    
