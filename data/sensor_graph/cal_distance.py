import os
import math
import geopy.distance

# calculate distance
def cal1():
    distance_file = open('distances.csv', 'w')
    distance_file.write('from,to,cost\n')
    
    station_ids, lats, lons = [], [], []
    with open('pm10_stations.csv', 'r') as f:
        for row in f:
            cols = row.split(',')
            station_ids.append(cols[1])
            lons.append(float(cols[2]))
            lats.append(float(cols[3]))
    for i in range(len(station_ids)):
        for j in range(len(station_ids)):
            s_from = station_ids[i]
            s_to = station_ids[j]
            cord1 = (lats[i], lons[i])
            cord2 = (lats[j], lons[j])
            d = int(geopy.distance.distance(cord1, cord2).km * 1000)
            distance_file.write(s_from + ',' + s_to + ',' + str(d) + '\n')
    distance_file.flush()
    distance_file.close()

def cal2():
    distance_file = open('distances_grid.csv', 'w')
    distance_file.write('from,to,cost\n')
    for i in range(1024):
        for j in range(1024):
            s_from = str(i)
            s_to = str(j)
            x = (int(i/32), i - int(i/32)*32)
            y = (int(j/32), j - int(j/32)*32)
            d = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
            distance_file.write(s_from + ',' + s_to + ',' + str(d) + '\n')
    distance_file.flush()
    distance_file.close()

def cal3():
    distance_file = open('distances_grid_pm10.csv', 'w')
    distance_file.write('from,to,cost\n')
    stations_list = []
    with open('graph_sensor_ids_pm10_grid.txt') as f:
        for row in f:
            stations_list = row.split(',')
    print(stations_list)
    for i in range(len(stations_list)):
        for j in range(len(stations_list)):
            s_from = int(stations_list[i])
            s_to = int(stations_list[j])
            x = (int(s_from/32), s_from - int(s_from/32)*32)
            y = (int(s_to/32), s_to - int(s_to/32)*32)
            d = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
            distance_file.write(str(s_from) + ',' + str(s_to) + ',' + str(d) + '\n')
    distance_file.flush()
    distance_file.close()

def cal4():
    distance_file = open('distances_grid_traffic.csv', 'w')
    distance_file.write('from,to,cost\n')
    pollution_stations = []
    with open('graph_sensor_ids_traffic.txt') as f:
        rows = []
        for row in f:
            rows.append(row.strip())

    pollution_stations = rows[0].split(',')
    print(pollution_stations)
    traffic_stations = rows[1].split(',')
    
    for i in range(len(pollution_stations)):
        # distances between pollution stations
        for j in range(len(pollution_stations)):
            s_from = int(pollution_stations[i])
            s_to = int(pollution_stations[j])
            x = (int(s_from/32), s_from - int(s_from/32)*32)
            y = (int(s_to/32), s_to - int(s_to/32)*32)
            d = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
            distance_file.write(str(s_from) + ',' + str(s_to) + ',' + str(d) + '\n')
    
        # distances with traffic stations
        for j in range(len(traffic_stations)):
            s_from = int(pollution_stations[i])
            s_to = int(traffic_stations[j])
            x = (int(s_from/32), s_from - int(s_from/32)*32)
            y = (int(s_to/32), s_to - int(s_to/32)*32)
            d = int(math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)])))
            distance_file.write(str(s_from) + ',' + str(s_to) + ',' + str(d) + '\n')
    distance_file.flush()
    distance_file.close()

cal4()

