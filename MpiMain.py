import collections, json, time
import io

from mpi4py import MPI
import numpy as np
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
# Credit stackoverflow
def merge(source, destination):
    for key, value in source.items():
        if isinstance(value, dict):
            # get node or create one
            node = destination.setdefault(key, collections.defaultdict(int))
            merge(value, node)
        else:
            destination[key] += value

    return destination

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)
start_time = 0
if rank == 0:
    start_time = time.time()


tweetFilePath = "bigTwitter.json"
try:
    tweetStream = io.open(tweetFilePath, "r", encoding="utf-8")
    gridsJson = json.load(open("sydGrid-2.json", "r"))
except IOError as e:
    print("Tweet file not found")
    quit()

processedGrids = [];
#get boundaries of each square
for grid in gridsJson['features']:
    boundary = np.array(grid['geometry']['coordinates'][0]).T
    leftRight = [min(boundary[0]), max(boundary[0])]
    topBottom = [max(boundary[1]), min(boundary[1])]
    processedGrids.append([leftRight, topBottom])

# print(processedGrids)

batch_size = 250
endOfFile= False

#Skip first row
next(tweetStream)
languageCount = collections.defaultdict(partial(collections.defaultdict, int))
while not endOfFile:
    # skip rows that are being processed by others
    for j in range(batch_size * rank):
        try:
            #if rank == 1: print(rank,j)
            next(tweetStream)
        except StopIteration:
            endOfFile = True
            break
            print("End of file")

    for i in range(batch_size):
        #print(next(tweetStream)[:-2])
        try:
            # print(rank,i)
            tweetStr = next(tweetStream)[:-2];
            tweetStr = tweetStr[:-1] if tweetStr[-1] == "]" else tweetStr
            try:
                tweet = json.loads(tweetStr)
                if tweet['doc']['coordinates']:
                    # print(tweet['doc']['coordinates']['coordinates'])
                    for boundary in processedGrids:
                        if isWithin(tweet['doc']['coordinates']['coordinates'], boundary, gridPosition[processedGrids.index(boundary)]):
                            languageCount[processedGrids.index(boundary)+1][tweet['doc']['lang']] += 1
            except json.decoder.JSONDecodeError:
                print("Unable to decode: ", tweetStr)
        except StopIteration:
            endOfFile = True
            print("End of file")
            break

    # skip rows that are being processed by others
    for j in range(batch_size * (comm.size - rank - 1)):
        try:
            next(tweetStream)
        except StopIteration:
            endOfFile = True
            break
            print("End of file")
print("Local result is ", json.dumps(languageCount))
languageCount = json.dumps(languageCount)
gatheredLanguageCount = comm.gather(languageCount,0)
if rank == 0:
    final_result= collections.defaultdict(partial(collections.defaultdict, int))
    # print(str(gatheredLanguageCount))
    for i in range(comm.size):
        final_result = merge(json.loads(gatheredLanguageCount[i]),final_result)
    print("Final result is: ",json.dumps(final_result))
    print("--- %s seconds ---" % (time.time() - start_time))