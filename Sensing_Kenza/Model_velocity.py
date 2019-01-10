##Step 1 - import all the necessary modules to carry out the different tasks

import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
import datetime
import math
import operator
from collections import Counter
import requests
import urllib.parse
import pandas as pd
from pymongo import MongoClient
from pprint import pprint
from statsmodels.tsa.arima_model import ARIMA

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
time_velocity2={}

for i in range(0,len(time_ms)):
    time_velocity2[time_ms2[i]]=velocities[i]

time_altitudes={}
for i in range(0,len(time_ms)):
    time_altitudes[time_ms2[i]]=altitudes[i]

time_ms2.sort()

    


time_ms2.sort()

time_ms3=[]
for i in range(0,len(time_ms2)):
    lop=str(time_ms2[i])
    #print(lop)
    time_ms3.append(lop)
    
velocities3=[]
for o in time_ms2:
    velocities3.append(time_velocity2[o])

altitudes_sorted=[]
for o in time_ms2:
    altitudes_sorted.append(time_altitudes[o])



velocitiess = np.array(velocities3)
altitudes_sorted2=np.array(altitudes_sorted)
col=["velocities","snow","temp","visi","precip","wind","altitudes"]
df = pd.DataFrame(columns=col,index = time_ms2, dtype=np.float64)
df.index.names = ['Date']
types=["snow","temp","visi","precip","wind"]
df['velocities']=velocitiess
df['altitudes']=altitudes_sorted2

print(df)
pd.set_option("display.max_rows",150)


df = df.resample('1H').max()
#df = df[np.isfinite(df['v'])] #remove NaN
#df = df.shift(+1)

count = 0
count_2=0
for i in df.index:
    if datetime.datetime(2018, 12, 16, 00, 1, 1) < i < datetime.datetime(2018, 12, 16, 23, 59, 59):
        df = df.drop(df.index[count_2])
    else:
        count_2=count_2+1


count_2=0
for i in df.index:
    if datetime.datetime(2018, 12, 29, 00, 00, 00) <= i <= datetime.datetime(2018, 12, 29, 23, 59, 59):
        df = df.drop(df.index[count_2])
    else:
        count_2=count_2+1

count_2=0
for i in df.index:
    if datetime.datetime(2018, 12, 22, 00, 00, 00) <= i < datetime.datetime(2018, 12, 27, 23, 59, 59):
        df = df.drop(df.index[count_2])
    else:
        count_2=count_2+1

df.loc[pd.to_datetime('2018-12-30 23:00:00')]=[0.0,0.0,-5,9,0.0,8,1055]

print(count)
print(count_2)

print(df)

df = df.fillna(0)

print(df)   
index = df.index


index_day = [['day1'],['day2'],['day3'],['day4'],['day5'],['day6'],['day7']]
days_2 = ['17','18','19','20','21','28','30']

date_ms5=[]

for x in range(0,len(index)):
    date_ms5.append(str(index[x])[8:10])


for i in range(0,len(date_ms5)):
    for j in range(0,7):
        if date_ms5[i] == days_2[j]:
            index_day[j].append(index[i])

ngf = []
list_1=[]
list_4=['Brevent-Flegere','Courmayeur','Brevent-Flegere','St Gervais Les Bains','Grands Montets','Brevent-Flegere','Contamines-Montjoie']
w = Weather()

index_type=[["snow"],["temp"],["visi"],["precip"], ["wind"]]


for h in range(0,7):
    query = w.get_location(list_4[h],"snow","temp","visi","precip", "wind")
    lok4 = (index_day[h][1:len(index_day[h])])
    for d in lok4:
        lod= str(d)[0:16]
        for g in range(0,len(query)):
            for k in query:
                if str(k) == types[g]:
                    index_type[g].append(query[str(k)][lod])

snows=[]
snows = np.array(index_type[0][1:len(index_type[0])])
df['snow']= snows
df['temp']= (index_type[1][1:len(index_type[1])])
df['visi']= (index_type[2][1:len(index_type[2])])
df['precip']= (index_type[3][1:len(index_type[3])])
df['wind']= (index_type[4][1:len(index_type[4])])




print(df.dtypes)
print(df)


export_df = df['velocities']
timestamps_velocities_updated = export_df.to_dict()


export_df2 = df['altitudes']
timestamps_altitudes_updated = export_df2.to_dict()

velocities_updated=[]



for i in timestamps_velocities_updated.keys():
    velocities_updated.append(timestamps_velocities_updated[i])

altitudes_updated=[]

for i in timestamps_altitudes_updated.keys():
    altitudes_updated.append(timestamps_altitudes_updated[i])


time_stamps_updated2=[]
for i in timestamps_velocities_updated.keys():
    d = i.timestamp() * 1000
    d= d*0.001
    d = datetime.datetime.fromtimestamp(d).isoformat()
    d = d.replace('T',' ')
    d= str(d)[0:16]
    time_stamps_updated2.append(str(d))

timestamps_velocities_updated2={}
for i in range(0,len(velocities_updated)):
    timestamps_velocities_updated2[time_stamps_updated2[i]]=velocities_updated[i]

timestamps_altitudes_updated2={}
for i in range(0,len(velocities_updated)):
    timestamps_altitudes_updated2[time_stamps_updated2[i]]=altitudes_updated[i]

               
class User:

    """
    A class to add, retrieve and delete user data from/to MongoDB Atlas.
    """

    def __init__(self):

        self.client = MongoClient("mongodb+srv://clarissebret:sensing-and-iot@cluster0-xrhtq.mongodb.net/")
        self.db = self.client.ski
        self.user = self.db.user

    @staticmethod
    def weather_query(date, q):

        """
        A static method retrieving weather data from World Weather Online local weather API.
        :param date: a date, either 'today', 'tomorrow' or in string format %Y-%m-%d
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

        params = {'key': key, 'q': q, "date": date, "tp": tp, "format": "json","includelocation":"yes"}
        query = urllib.parse.urlencode(params)
        url_query = urllib.parse.unquote(query)
        url = 'http://api.worldweatheronline.com/premium/v1/weather.ashx?' + url_query
        response = requests.get(url)
        data = response.json()
        weather = data["data"]["weather"][0]

        for i in range(0, len(weather["hourly"])):
            t = weather['date'] + " " + str(int(int(weather["hourly"][i]["time"]) * 0.01))
            ts = datetime.datetime.strptime(t, "%Y-%m-%d %H").strftime("%Y-%m-%d %H:%M")
            temp[ts] = weather["hourly"][i]["tempC"]
            visi[ts] = weather["hourly"][i]["cloudcover"]
            precip[ts] = weather["hourly"][i]["precipMM"]
            wind[ts] = weather["hourly"][i]["windspeedKmph"]
            snow[ts] = weather["totalSnow_cm"]

        return temp, visi, precip, wind, snow

    def get_user(self, name):

        query = list(self.user.find(
            {"_id": name},
            {"_id": 0}
        ))

        if len(query) == 0:
            print("%s is not in the database. Use add_location to insert it." % name)
        else:
            return query[0]

    def add_user(self, name, data):

        data["_id"] = name
        query = list(self.user.find({"_id": name}))

        if len(query) == 0:
            self.user.insert_one(data)
        else:
            print("%s was already added to this collection. Use update_user instead." % name)

    def update_user(self, user, object, data):

        self.user.update_one({"_id": user}, {"$set": { object: data }})

    def delete_user(self, user):

        query = list(self.user.find({"_id": user}))

        if len(query) == 0:
            print("%s is not in this collection." % user)

        else:
            self.user.delete_one({"_id":user})

if __name__ == '__main__':

    user = User()

    name = "Kenza" # user name in string format
    param = "altitude"  # parameter in string format
    data = {"ts":"value"} # data in dic format

    user.update_user(name, param, timestamps_altitudes_updated2)

df2 = df.loc[df['velocities'] < 19]
df2 = df.loc[df['velocities'] > 0]



print(df2)



dhu = df2['temp']
dhj=[]

for i in dhu:
    float(i)
    dhj.append(i)
    
a = np.array(dhj, dtype=np.float32)
print (a)

#print(df.dtype)


#df = df.convert_objects(convert_numeric=True).dtype
#print(df.dtype)
#df = df.infer_objects(0).dtypes
#print(df.dtypes)





b = np.arange(1, 18, 0.5)

model = ARIMA(endog=df2['velocities'], order=(1,1,1), exog=dhj)
#model = ARIMA(endog=df['velocities'], order=(5,1,0), exog=df.iloc[:, 1:5])
model_fit = model.fit()
print(model_fit.summary())

#forecast = model_fit.forecast()[0]

#model_fit.plot_diagnostics(figsize=(16, 8))
#plt.show()

forecast = model_fit.predict(start=5, end=10)
#forecast = model_fit.forecast(steps=7)[0]


print(forecast)

#model.plot_predict(h=10, oos_data=b, past_values=100, figsize=(15,5))
