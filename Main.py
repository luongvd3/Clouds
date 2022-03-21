import collections
from mpi4py import MPI
import numpy as np
import json
from functools import partial

gridPosition = ["topLeft", "left", "left", "bottomLeft","top","center","center","bottom","top","center","center","bottom","topRight","right","right","bottomRight"]
# decide if a coordinates is within a region following all the conditions set out in the assignment specs
# note that the grids coordinates have more numbers behind the floating point or  are more details than the ones from the tweets so it's not
# possible a tweet to fall on a border of a grid, so the extra conditions seem unnecessary
def isWithin(point, boundary, position):
    if position == "center" or position == "right" or position == "top" or position == "topRight":
        if boundary[0][0] < point[0] < boundary[0][1] and boundary[1][1] < point[1] < boundary[1][0]:
            return True
        elif point[0] == boundary[0][1] and boundary[1][1] <= point[1] < boundary[1][0]:
            return True
        elif boundary[0][0] < point[0] <= boundary[0][1] and boundary[1][0] == point[1]:
            return True
    elif position == "left" or position == "topLeft":
        if boundary[0][0] < point[0] < boundary[0][1] and boundary[1][1] < point[1] < boundary[1][0]:
            return True
        elif (point[0] == boundary[0][1] or point[0] == boundary[0][0]) and boundary[1][1] <= point[1] < boundary[1][0]:
            return True
        elif boundary[0][0] < point[0] <= boundary[0][1] and boundary[1][0] == point[1]:
            return True
    elif position == "bottom" or position == "bottomRight":
        if boundary[0][0] < point[0] < boundary[0][1] and boundary[1][1] < point[1] < boundary[1][0]:
            return True
        elif point[0] == boundary[0][1] and boundary[1][1] <= point[1] < boundary[1][0]:
            return True
        elif boundary[0][0] < point[0] <= boundary[0][1] and (boundary[1][0] == point[1] or boundary[1][1] == point[1]):
            return True
    elif position == "bottomLeft":
        if boundary[0][0] < point[0] < boundary[0][1] and boundary[1][1] < point[1] < boundary[1][0]:
            return True
        elif (point[0] == boundary[0][1] or point[0] == boundary[0][0]) and boundary[1][1] <= point[1] < boundary[1][0]:
            return True
        elif boundary[0][0] < point[0] <= boundary[0][1] and (boundary[1][0] == point[1] or boundary[1][1] == point[1]):
            return True
    else:
        return False

def merge(source, destination):
    """
    run me with nosetests --with-doctest file.py

    True
    """
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, collections.defaultdict(int))
            merge(value, node)
        else:
            destination[key] += value

    return destination

tweetFilePath = "smallTwitter.json"
try:
    tweetStream = open("smallTwitter.json", "r", encoding="utf-8")
    gridsJson = json.load(open("sydGrid-2.json", "r"))
except FileNotFoundError as e:
    print("Tweet file not found")
    quit()

print(gridsJson['features'][0]['geometry']['coordinates'][0])

processedGrids = [];
#get boundaries of each square
for grid in gridsJson['features']:
    leftRight = [grid['geometry']['coordinates'][0][1][0],grid['geometry']['coordinates'][0][2][0]]
    topBottom = [grid['geometry']['coordinates'][0][0][1],grid['geometry']['coordinates'][0][1][1]]
    processedGrids.append([leftRight, topBottom])

print(processedGrids)
#Skip first row
next(tweetStream)
languageCount = collections.defaultdict(partial(collections.defaultdict, int))
for i in range(5000):
    #print(next(tweetStream)[:-2])
    try:
        tweetStr = next(tweetStream)[:-2];
        tweetStr = tweetStr[:-1] if tweetStr[-1] == "]" else tweetStr
        tweet = json.loads(tweetStr)
        #print(tweet)
        #print(tweet['doc']['metadata']['iso_language_code'])
        if tweet['doc']['coordinates']:
            print(tweet['doc']['coordinates']['coordinates'])
            for boundary in processedGrids:
                if isWithin(tweet['doc']['coordinates']['coordinates'], boundary, gridPosition[processedGrids.index(boundary)]):
                    languageCount[processedGrids.index(boundary)+1][tweet['doc']['metadata']['iso_language_code']] += 1
    except StopIteration:
        print("End of file")

#Test
for pair in processedGrids:
    assert pair[0][0] < pair[0][1]
    assert pair[1][0] > pair[1][1]
a = collections.defaultdict(partial(collections.defaultdict, int))
a['1']['es'] = 10
# languageCount['1']['es'] += 10
print(merge(a,languageCount))