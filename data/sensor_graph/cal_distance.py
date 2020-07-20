import os
import math
import geopy.distance

newfile = open('distances_pm10.csv', 'w')
newfile.write('from,to,cost\n')
station_ids, lats, lons = [], [], []
with open('pm10_stations.csv', 'r') as f:
    for row in f:
        cols = row.split(',')
        station_ids.append(cols[1])
        lons.append(float(cols[2]))
        lats.append(float(cols[3]))

# calculate distance
def cal1():
    for i in range(len(station_ids)):
        for j in range(len(station_ids)):
            s_from = station_ids[i]
            s_to = station_ids[j]
            cord1 = (lats[i], lons[i])
            cord2 = (lats[j], lons[j])
            d = int(geopy.distance.distance(cord1, cord2).km * 1000)
            newfile.write(s_from + ',' + s_to + ',' + str(d) + '\n')

def cal2():
    for i in range(1024):
        for j in range(1024):
            s_from = str(i)
            s_to = str(j)
            x = (int(i/32), i - int(i/32)*32)
            y = (int(j/32), j - int(j/32)*32)
            d = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
            newfile.write(s_from + ',' + s_to + ',' + str(d) + '\n')
cal1()
newfile.flush()
newfile.close()
