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

resorts = {
    'Brevent-Flegere': '45.934,6.839',
    'Balme': '46.042,6.952',
    'Grands Montets': '45.957,6.952',
    'Houches': '45.885,6.752',
    'Courmayeur': '45.790,6.933',
    'Verbiers': '46.091,7.254',
    'St Gervais Les Bains': '45.849,6.614',
    'Contamines-Montjoie': '45.961,6.887'
}

dates = ['2018-12-17', '2018-12-18', '2018-12-19', '2018-12-20', '2018-12-21', '2018-12-22', '2018-12-23',
         '2018-12-24', '2018-12-25', '2018-12-26', '2018-12-27', '2018-12-28', '2018-12-29', '2018-12-30']

class Weather:

    """
    A class to add, retrieve and delete past weather data from/to MongoDB Atlas.
    """

    def __init__(self):

        self.client = MongoClient("mongodb+srv://clarissebret:sensing-and-iot@cluster0-xrhtq.mongodb.net/")
        self.db = self.client.ski
        self.weather = self.db.weather

    @staticmethod
    def weather_query(date, q):

        """
        A static method retrieving weather data from World Weather Online local historical weather API.
        :param date: a list of dates in string format %Y-%m-%d
        :param q: geographical coordinates in decimal string format XXX.XXX, XXX.XXX
        :return: 5 time series: temperature (ºC), precipitations (mm), wind (km/h), snow falls (cm)
        """

        key = '483a51a481274ca2a74223116182612'
        tp = 1

        temp = {}
        visi = {}
        precip = {}
        wind = {}
        snow = {}

        for j in range(0, len(date)):

            params = {'key': key, 'q': q, "date": date[j], "tp": tp, "format": "json","includelocation":"yes"}
            query = urllib.parse.urlencode(params)
            url_query = urllib.parse.unquote(query)
            url = 'http://api.worldweatheronline.com/premium/v1/past-weather.ashx?' + url_query
            response = requests.get(url)
            data = response.json()
            weather = data["data"]["weather"][0]

            for i in range(0, len(weather["hourly"])):
                t = weather['date'] + " " + str(int(int(weather["hourly"][i]["time"]) * 0.01))
                ts = datetime.strptime(t, "%Y-%m-%d %H").strftime("%Y-%m-%d %H:%M")
                temp[ts] = weather["hourly"][i]["tempC"]
                visi[ts] = weather["hourly"][i]["cloudcover"]
                precip[ts] = weather["hourly"][i]["precipMM"]
                wind[ts] = weather["hourly"][i]["windspeedKmph"]
                snow[ts] = weather["totalSnow_cm"]

        return temp, visi, precip, wind, snow

    def get_location(self, place, name = "", q = "", temp = "", visi = "", precip = "", wind = "", snow = ""):

        query = list(self.weather.find(
            {"name": place},
            {"_id": 0, name: 1, q: 1, temp: 1, visi: 1, precip: 1, wind: 1, snow: 1}
        ))

        if len(query) == 0:
            print("%s is not in the database. Use add_location to insert it." % place)

        else:

            return query[0]

    def add_location(self, date, place, q):

        query = list(self.weather.find({"name": place}))

        if len(query) == 0:

            location = {}
            weather = self.weather_query(date, q)
            location["name"] = place
            location["q"] = q
            location["temp"] = weather[0]
            location["visi"] = weather[1]
            location["precip"] = weather[2]
            location["wind"] = weather[3]
            location["snow"] = weather[4]
            self.weather.insert_one(location)

        else:

            print("%s was already added to this collection: use update_location instead." % place)

    def delete_location(self, place):

        query = list(self.weather.find({"name": place}))

        if len(query) == 0:
            print("%s is not in the database. Use add_location to insert it." % place)

        else:
            self.weather.delete_one({"name":place})

#if __name__ == '__main__':


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

#plt.plot(altitudes)
#plt.show()


#plt.plot(latitudes,longitudes)
#plt.show()

#plt.plot(stamp_ms,altitudes)
#plt.show()

#plt.plot(velocities)
#plt.show()

    
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

#print(time_velocity)

#print (time_stamp.get('2018-12-23T16:10:47', "none"))


#print(coordinate_update[3][1:len(coordinate_update[3])])


#apply the functions on the relevant lists
    
nbofitems = [7,35,7,10,15]
nbofitems_2=[15,10,2,35,7]
testInstances = [[45.9665372, 6.9433641999999995],[45.849677199999995, 6.614935699999999],[45.941119199999996, 6.854474499999999]]
testInstances_2 =[[45.961214399999996, 6.8869466],[45.803365799999995, 6.9349422],[45.941119199999996, 6.854474499999999],[45.849677199999995, 6.614935699999999],[45.9665372, 6.9433641999999995]]
timess=[]
durations=[]
#print(len(nbofitems_2))
#print(len(testInstances_2))
#print(len(coordinate_update))
#print(len(coordinates_day))
#print(coordinates_day[0])
#coordinates_day.remove('day11')
#print(coordinates_day.pop([7]))
#print(coordinates_day.pop([8]))
#print(coordinates_day.pop([9]))
#print(len(coordinates_day))

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

#print(timess)
#print(durations)

#print(time_ms2)
time_ms2.sort()

velocities3=[]

for o in time_ms2:
    #print (o, time_velocity[o])
    velocities3.append(time_velocity2[o])
    


time_ms3=[]
for i in range(0,len(time_ms2)):
    lop=str(time_ms2[i])
    #print(lop)
    time_ms3.append(lop)

#print(type(time_ms3))
#print(time_ms2)
velocitiess = np.array(velocities3)
df = pd.DataFrame(velocitiess,columns=list('v'), index = time_ms2)
df.index.names = ['Date']

#print(df)

df = df.resample('1H').max()
df = df[np.isfinite(df['v'])] #remove NaN

#print(df)

df = df.loc[df['v'] < 19]
df = df.loc[df['v'] > 0]
df = df.drop(df.index[0])
df = df.drop(df.index[30])
df = df.drop(df.index[30])
df = df.drop(df.index[30])

#print(df)

#conditions = json.loads(open('activity_clarisse2.json').read())
#conditions = json.loads(json.dumps(weather_data-2.json"))
#conditions = open("weather_data-2.rtf","w")
#print(conditions)
#brevent={}
#snow=[]
#tempera=[]
##Step 4 - append each type of data to a separate list to be easily used
#for x in range(0, len(conditions)):
    #brevent = (conditions[x][ 'Brevent'])
    #tempera.append(conditions[x]['temp'])
    #snow.append(conditions[x]['snow'])
#print(brevent)




    
