import collections
import io
import time

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

    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, collections.defaultdict(int))
            merge(value, node)
        else:
            destination[key] += value

    return destination
start_time = time.time()
tweetFilePath = "bigTwitter.json"
try:
    tweetStream = io.open(tweetFilePath, "r", encoding="utf-8")
    gridsJson = json.load(open("sydGrid-2.json", "r"))
    languages = [line[:-1].split(" ") for line in open("languages.txt", "r", encoding="utf-8").readlines()]
except IOError as e:
    print("Tweet file not found")
    quit()

print(gridsJson['features'][0]['geometry']['coordinates'][0])

processedGrids = [];
#get boundaries of each square
for grid in gridsJson['features']:
    boundary = np.array(grid['geometry']['coordinates'][0]).T
    leftRight = [min(boundary[0]), max(boundary[0])]
    topBottom = [max(boundary[1]), min(boundary[1])]
    processedGrids.append([leftRight, topBottom])
languagesDict = {}
for line in languages:
    languagesDict[line[1]] = line[0]
print(languagesDict)
print(processedGrids)
#Skip first row
next(tweetStream)
languageCount = collections.defaultdict(partial(collections.defaultdict, int))
for i in range(5000):
    #print(next(tweetStream)[:-2])
    try:
        tweetStr = next(tweetStream)[:-2];
        tweetStr = tweetStr[:-1] if tweetStr[-1] == "]" else tweetStr
        try:
            tweet = json.loads(tweetStr)
            if tweet['doc']['coordinates'] and tweet['doc']['lang'] != "und" and tweet['doc']['lang'] != "null":
                # print(tweet['doc']['coordinates']['coordinates'])
                for boundary in processedGrids:
                    if isWithin(tweet['doc']['coordinates']['coordinates'], boundary, gridPosition[processedGrids.index(boundary)]):
                        if tweet['doc']['lang'] in languagesDict.keys():
                            languageCount[processedGrids.index(boundary)+1][languagesDict[tweet['doc']['lang']]] += 1
                        else:
                            languageCount[processedGrids.index(boundary)+1][tweet['doc']['lang']] += 1

        except json.decoder.JSONDecodeError:
            print(tweetStr)
        #print(tweet)
        #print(tweet['doc']['metadata']['iso_language_code'])
    except StopIteration:
        print("End of file")
        break
print("--- %s seconds ---" % (time.time() - start_time))
languageCount = dict(sorted(languageCount.items(), key=lambda item: len(item[1])))
final_result = {}
for region, data in languageCount.items():
    data = dict(sorted(data.items(), key=lambda item: item[1]))
    final_result[region] = {}
    final_result[region]["Total Tweet"] = sum(data.values())
    final_result[region]["Number of Languages Used"] = len(data)
    top10 = ""
    i = 0
    while i < len(data) and i < 10:
        top10 += " "+ list(data.keys())[i] + "-" + str(list(data.values())[i])
        i += 1
    final_result[region]["Top 10 Languages & #Tweets"] = top10
#Test
for pair in processedGrids:
    assert pair[0][0] < pair[0][1]
    assert pair[1][0] > pair[1][1]
a = collections.defaultdict(partial(collections.defaultdict, int))
a['1']['es'] = 10
# languageCount['1']['es'] += 10
print(merge(a,languageCount))
print(final_result)
