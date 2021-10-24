#!/usr/bin/env python
import numpy
import math
from queue import *
import matplotlib.pyplot as plt
import matplotlib
import unittest
import random


#Vertex Class
class vertex(object):
  def __init__(self, neighbors, neighborsCost, x):
    self.x = x
    self.neighbors = neighbors
    self.neighborsCost = neighborsCost
    self.g = None
    self.backpointers = None


class priorityVertex(object):
  def __init__(self, priority, node):
    self.priority = priority
    self.node = node

  def __lt__(self, other):
    return self.priority < other.priority

#Grid Class - Grid that is then fed into grid2graph to create the graph
class grids(object):
  def __init__(self, xx, yy, F):
    self.xx = xx
    self.yy = yy
    self.F = F

#Plots the graph as a series of print statements
def graph_plot(vector):
  maxX = 0;
  maxY = 0;
  #Finds the dimensions of the needed array to print
  for node in vector:
    if node.x[0][0] > maxX:
      maxX = node.x[0][0]
    if node.x[1][0] > maxY:
      maxY = node.x[1][0]
  #Makes the empty array
  graph = numpy.empty(shape=(maxX*2-1,maxY*2-1),dtype='object')

  #Fills the array with spaces
  for k in range(maxX*2-1):
    for j in range(maxY*2-1):
      graph[k][j] = '   '
  
  i = 1
  for node in vector:
    #Adds in the nodes in their correct locations
    if i < 10:
      graph[(node.x[0][0]-1)*2][(node.x[1][0]-1)*2] = '00%s' % i
    else:
      graph[(node.x[0][0]-1)*2][(node.x[1][0]-1)*2] =  '0%s' % i
    i += 1
    
    #Adds the edges between neighbors
    for neighborNum in node.neighbors:
      neighbor = vector[neighborNum[0]-1]
      #Finds the space to put the marker
      avgX = ((neighbor.x[0][0]-1)*2 + (node.x[0][0]-1)*2)/2
      avgY = ((neighbor.x[1][0]-1) + (node.x[1][0]-1))
      #Determines which markers to use
      if(avgY % 2 == 1 and avgX % 2 == 1):
        if(neighbor.x[0][0] > node.x[0][0]):
          if(neighbor.x[1][0] > node.x[1][0]):
            if(graph[int(avgX)][int(avgY)] == ' / '):
              graph[int(avgX)][int(avgY)] = ' X '
            else:
              graph[int(avgX)][int(avgY)] = ' \\ '
          else:
            if(graph[int(avgX)][int(avgY)] == ' \\ '):
              graph[int(avgX)][int(avgY)] = ' X '
            else:
              graph[int(avgX)][int(avgY)] = ' / '
        else:
          if(neighbor.x[1][0] > node.x[1][0]):
            if(graph[int(avgX)][int(avgY)] == ' \\ '):
              graph[int(avgX)][int(avgY)] = ' X '
            else:
              graph[int(avgX)][int(avgY)] = ' / '
          else:
            if(graph[int(avgX)][int(avgY)] == ' / '):
              graph[int(avgX)][int(avgY)] = ' X '
            else:
              graph[int(avgX)][int(avgY)] = ' \\ '
      elif(avgY % 2 == 1):
        graph[int(avgX)][int(avgY)] = ' - '
      else:
        graph[int(avgX)][int(avgY)] = ' | '

  #Prints out the array representing the graph
  for k in range(maxX*2-1):
    for j in range(maxY*2-1):
      print('%s' % graph[k][j], end ="")
    print('')

#Plots a graph using matplotlib 
def matPlotGraph(vector):
  xpoints = numpy.empty(shape=(len(vector),1),dtype='object')
  ypoints = numpy.empty(shape=(len(vector),1),dtype='object')
  count = 0
  maxX = 0
  maxY = 0
  for node in vector:
    plt.annotate(str(count),(node.x[0][0], node.x[1][0]))
    xpoints[count] = node.x[0]
    ypoints[count] = node.x[1]
    if node.x[0][0] > maxX:
      maxX = node.x[0][0]
    if node.x[1][0] > maxY:
      maxY = node.x[1][0]

    for neighbor in node.neighbors:
      if (neighbor[0]-1) > count:
        plt.plot(numpy.array([node.x[0], vector[neighbor[0]-1].x[0]]), numpy.array([node.x[1], vector[neighbor[0]-1].x[1]]), 'b')
    count += 1

  plt.ion()
  plt.scatter(xpoints, ypoints)
  plt.xlim([0, maxX+1])
  plt.ylim([0, maxY+1])
  
  plt.show()

  
#Finds the nearest neighbors to a given point  
def graph_nearestNeighbors(vector, x, k):
  q = PriorityQueue()
  i = 1
  #Finds the distance to each neighbor in the vectos
  for neighbor in vector:
    distance = math.sqrt((neighbor.x[0][0]-x[0][0])**2 + (neighbor.x[1][0]-x[1][0])**2)
    q.put([distance, i])
    i += 1
  
  idxNeighbors = []
  j = 0
  #Returns the wanted number of closest neighbors
  while j < k and not q.empty():
    idxNeighbors.append(q.get()[1])
    j += 1
  return idxNeighbors

#Turns a grid into a graph
def grid2graph(grid):

  #Initializes the graph vector and an array to keep track of indexes
  graphVector = []
  idxGrid = numpy.empty(shape=(len(grid.xx), len(grid.yy)), dtype = 'object')
  
  
  count = 1
  i = 0
  #For every point in the grid, checks if the vertex exists
  for row in grid.xx:
    j = 0
    for col in grid.yy:
      if grid.F[i][j]:
        
        tempNeighbors = []
        tempCosts = []
        idxGrid[i][j] = count
        #Checks if square above current exists
        if i > 0 and grid.F[i-1][j]:
          idx = idxGrid[i-1][j]
          cost = row-grid.xx[i-1]
          tempNeighbors.append([idx])
          tempCosts.append([cost])
          if graphVector[idx-1].neighbors == []:
            graphVector[idx-1].neighbors = [[count]]
            graphVector[idx-1].neighborsCost = [[cost]]
          else:
            graphVector[idx-1].neighbors.append([count])
            graphVector[idx-1].neighborsCost.append([cost])
        #Checks if Square Up Left of current Exists
        if i > 0 and j > 0 and grid.F[i-1][j-1]:
          idx = idxGrid[i-1][j-1]
          cost = math.sqrt((row-grid.xx[i-1])**2 + (col-grid.yy[j-1])**2)
          tempNeighbors.append([idx])
          tempCosts.append([cost])
          if graphVector[idx-1].neighbors == []:
            graphVector[idx-1].neighbors = [[count]]
            graphVector[idx-1].neighborsCost = [[cost]]
          else:
            graphVector[idx-1].neighbors.append([count])
            graphVector[idx-1].neighborsCost.append([cost])
        #Checks if Square UP Right of current Exists
        if i > 0 and j < (len(grid.yy) - 1) and grid.F[i-1][j+1]:
          idx = idxGrid[i-1][j+1]
          cost = math.sqrt((row-grid.xx[i-1])**2 + (col-grid.yy[j+1])**2)
          tempNeighbors.append([idx])
          tempCosts.append([cost])
          if graphVector[idx-1].neighbors == []:
            graphVector[idx-1].neighbors = [[count]]
            graphVector[idx-1].neighborsCost = [[cost]]
          else:
            graphVector[idx-1].neighbors.append([count])
            graphVector[idx-1].neighborsCost.append([cost])
        #Checks if Square Left of current Exists
        if j > 0 and grid.F[i][j-1]:
          idx = idxGrid[i][j-1]
          cost = col-grid.yy[j-1]
          tempNeighbors.append([idx])
          tempCosts.append([cost])
          if graphVector[idx-1].neighbors == []:
            graphVector[idx-1].neighbors = [[count]]
            graphVector[idx-1].neighborsCost = [[cost]]
          else:
            graphVector[idx-1].neighbors.append([count])
            graphVector[idx-1].neighborsCost.append([cost])
        
        #Adds found neighbors to current point
        graphVector.append(vertex(tempNeighbors, tempCosts, [[row], [col]]))
        count += 1
      j += 1
    i += 1 

  return graphVector

 
#Return the heuristic difference between vertices with indexes idxX and idxGoal
def graph_heuristic(graphVector, idxX, idxGoal):
  coordCurrent = graphVector[idxX].x
  coordWanted = graphVector[idxGoal].x
  #Distance Formula
  hVal = math.sqrt((coordWanted[0][0]-coordCurrent[0][0])**2 + (coordWanted[1][0]-coordCurrent[1][0])**2)
  return hVal

#Figures out which neighbors haven't been checked yet
def graph_getExpandList(graphVector, idxNBest, idxClosed):
  vertex = graphVector[idxNBest]
  retList = []
  for i in range(len(vertex.neighbors)):
    if not (vertex.neighbors[i][0]-1) == idxNBest:
      if not (vertex.neighbors[i][0]-1) in idxClosed:
        retList.append(vertex.neighbors[i][0]-1)
  return retList

#Figures out whether the elements need to be added to the queue and updates the distance to the start and the backpointer if needed
def graph_expandElement(graphVector, idxNBest, idxX, idxGoal, pqOpen):
  i = 0
  for neighbor in graphVector[idxNBest].neighbors:
    if neighbor[0] == idxX+1:
      break
    i += 1
  cost = graphVector[idxNBest].neighborsCost[i][0]
  g = graphVector[idxNBest].g + cost
  newQ = PriorityQueue()
  qArray = []

  while not pqOpen.empty():
    temp = pqOpen.get()
    qArray.append(temp)
    newQ.put(temp)

  isIn = False
  for item in qArray:
    if item.node == idxX:
      isIn = True
      break

  if not isIn:
    graphVector[idxX].g = g
    graphVector[idxX].backpointers = idxNBest
    fn = g + graph_heuristic(graphVector, idxX, idxGoal)
    newQ.put(priorityVertex(fn, idxX))
  elif g < graphVector[idxX].g:
    graphVector[idxX].g = g
    graphVector[idxX].backpointers = idxNBest
  return [graphVector, newQ]

#A* Program to find the ideal path
def graph_path(graphVector, idxStart, idxGoal):
  currIdx = idxGoal
  xPath = []
  while not currIdx == idxStart:
    xPath.append(graphVector[currIdx].x)
    currIdx = graphVector[currIdx].backpointers 
  xPath.append(graphVector[idxStart].x)
  xPath = [ele for ele in reversed(xPath)]
  return xPath

def graph_search(graphVector, idxStart, idxGoal):
  O = PriorityQueue()
  C = []
  nStart = graphVector[idxStart]
  nStart.g = 0
  O.put(priorityVertex(0, idxStart))
  while not O.empty():
    nBestTemp = O.get()
    nBest = nBestTemp.node
    C.append(nBest)
    if nBest == idxGoal:
      break
    listToAdd = graph_getExpandList(graphVector, nBest, C)
    for x in listToAdd:
      temp = graph_expandElement(graphVector, nBest, x, idxGoal, O)
      graphVector = temp[0]
      O = temp[1]
  xPath = graph_path(graphVector, idxStart, idxGoal)
  return [xPath, graphVector]

def updateGraph(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector):
  if len(xpoints) > 50:
    plt.pause(0.01)
  else:
    plt.pause(0.5)
  plt.clf()
  for i in range(len(xpoints)):
    if not colours[i] == 'w':
      plt.scatter(xpoints[i], ypoints[i], c = colours[i], marker = shapes[i], zorder = 2)
  plt.xlim([0, maxX+1])
  plt.ylim([0, maxY+1])
  
  count = 0
  for node in graphVector:
    for neighbor in node.neighbors:
      if (neighbor[0]-1) > count:
        if not colours[count] == 'w' and not colours[neighbor[0]-1] == 'w':
          plt.plot(numpy.array([node.x[0], graphVector[neighbor[0]-1].x[0]]), numpy.array([node.x[1], graphVector[neighbor[0]-1].x[1]]), 'y', zorder = 1)
    count += 1

  plt.show()

def updateGraphFinal(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector, xPath):
  if len(xpoints) > 50:
    plt.pause(0.01)
  else:
    plt.pause(0.5)
  plt.clf()
  
  for i in range(len(xpoints)):
    if not colours[i] == 'w':
      plt.scatter(xpoints[i], ypoints[i], c = colours[i], marker = shapes[i], zorder = 2)

  plt.xlim([0, maxX+1])
  plt.ylim([0, maxY+1])
  
  count = 0
  for node in graphVector:
    for neighbor in node.neighbors:
      if (neighbor[0]-1) > count:
        if not colours[count] == 'w' and not colours[neighbor[0]-1] == 'w':
          plt.plot(numpy.array([node.x[0], graphVector[neighbor[0]-1].x[0]]), numpy.array([node.x[1], graphVector[neighbor[0]-1].x[1]]), 'y', zorder = 1)
    count += 1

  for i in range(len(xPath)-1):
    plt.plot(numpy.array([xPath[i][0], xPath[i+1][0]]), numpy.array([xPath[i][1], xPath[i+1][1]]), 'g', zorder = 1)

  plt.show(block = True)



def animated_graph_search(graphVector, idxStart, idxGoal):
  xpoints = numpy.empty(shape=(len(graphVector),1),dtype='object')
  ypoints = numpy.empty(shape=(len(graphVector),1),dtype='object')
  shapes = numpy.empty(len(graphVector), dtype='object')
  for i in range(0,len(graphVector)):
    shapes[i] = 'o'
  colours = [0]*len(graphVector)
  for i in range(0,len(graphVector)):
    colours[i] = 'w'
  plt.ion()
  count = 0
  maxX = 0
  maxY = 0
  for node in graphVector:
    xpoints[count] = node.x[0]
    ypoints[count] = node.x[1]
    if node.x[0][0] > maxX:
      maxX = node.x[0][0]
    if node.x[1][0] > maxY:
      maxY = node.x[1][0]
    count += 1
  
  updateGraph(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector)
  O = PriorityQueue()
  C = []
  nStart = graphVector[idxStart]
  nStart.g = 0
  O.put(priorityVertex(0, idxStart))
  colours[idxGoal] = 'r'
  shapes[idxGoal] = 'd'
  while not O.empty():
    for i in range(len(colours)):
      if colours[i] == 'g' and shapes[i] == 'x':
        if i == idxStart:
          colours[i] = 'r'
          shapes[i] = 'x'
        elif i in C:
          colours[i] = 'b'
          shapes[i] = 's'
        else:
          colours[i] = 'r'
          shapes[i] = 'o'
    
    nBestTemp = O.get()
    nBest = nBestTemp.node
    colours[nBest] = 'g'
    shapes[nBest] = 's'
    if nBest == idxStart:
      colours[nBest] = 'r'
      shapes[nBest] = 'x'
    C.append(nBest)
    if nBest == idxGoal:
      colours[nBest] = 'r'
      shapes[nBest] = 'd'
      colours[graphVector[nBest].backpointers] = 'b'
      shapes[graphVector[nBest].backpointers] = 's'
      break
    listToAdd = graph_getExpandList(graphVector, nBest, C)
    for x in listToAdd:
      temp = graph_expandElement(graphVector, nBest, x, idxGoal, O)
      graphVector = temp[0]
      O = temp[1]
      colours[x] = 'r'
      shapes[x] = 'o'
    for x in C:
      if not x == nBest and not x == idxStart:
        colours[x] = 'b'
        shapes[x] = 's'

    for i in range(len(graphVector[nBest].neighbors)):
      if not (graphVector[nBest].neighbors[i][0]-1) == nBest:
        colours[graphVector[nBest].neighbors[i][0]-1] = 'g'
        shapes[graphVector[nBest].neighbors[i][0]-1] = 'x'
    updateGraph(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector)
  for x in C:
      if not x == idxGoal and not x == idxStart:
        colours[x] = 'b'
        shapes[x] = 's'
  xPath = graph_path(graphVector, idxStart, idxGoal)
  updateGraph(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector)
  updateGraphFinal(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector, xPath)
  return [xPath, graphVector]

graphVectorMedium=[]
graphVectorMedium.append(vertex([[5], [2]], [[1],[1]], [[1],[1]]))
graphVectorMedium.append(vertex([[5], [3], [1]], [[1.4142],[1], [1]], [[1],[2]]))
graphVectorMedium.append(vertex([[6], [4], [2]], [[1.4142],[1], [1]], [[1],[3]]))
graphVectorMedium.append(vertex([[6], [3]], [[1],[1]], [[1],[4]]))
graphVectorMedium.append(vertex([[7], [2], [1]], [[1],[1.4142],[1]], [[2],[1]]))
graphVectorMedium.append(vertex([[8], [9], [4], [3]], [[1.4142],[1],[1],[1.4142]], [[2],[4]]))
graphVectorMedium.append(vertex([[10], [11], [5]], [[1],[1.4142], [1]], [[3],[1]]))
graphVectorMedium.append(vertex([[11], [9], [6]], [[1.4142],[1], [1.4142]], [[3],[3]]))
graphVectorMedium.append(vertex([[6], [8]], [[1],[1]], [[3],[4]]))
graphVectorMedium.append(vertex([[12], [13], [11], [7]], [[1],[1.4142], [1], [1]], [[4],[1]]))
graphVectorMedium.append(vertex([[12], [13], [14], [8], [7], [10]], [[1.4142],[1], [1.4142], [1.4142], [1.4142], [1]], [[4],[2]]))
graphVectorMedium.append(vertex([[13], [11], [10]], [[1],[1.4142],[1]], [[5],[1]]))
graphVectorMedium.append(vertex([[14], [11], [10], [12]], [[1],[1],[1.4142],[1]], [[5],[2]]))
graphVectorMedium.append(vertex([[15], [11], [13]], [[1], [1.4142],[1]], [[5],[3]]))
graphVectorMedium.append(vertex([[14]], [[1]], [[5],[4]]))

# creating graphVector #1
graphVector= []
graphVector.append(vertex([[2], [3]], [[1],[1]], [[1],[1]]))
graphVector.append(vertex([[1], [3]], [[1],[1.414]], [[2],[1]]))
graphVector.append(vertex([[1], [2], [4]], [[1],[1.414],[1]], [[1],[2]]))
graphVector.append(vertex([[3]], [2], [[1],[3]]))

#Plots the two vectors
graph_plot(graphVector)
print('---------------')
graph_plot(graphVectorMedium)

#Finds the nearest neighbor to a point - tests graph_nearestNeighbors
idxNeighbors = graph_nearestNeighbors(graphVectorMedium, [[0.78],[2.56]], 1)


#Creates a grid to be turned into a graph and then turns it into a graph and then plots it
grid = grids([1,2],  [1, 2, 3], [[True, True, True], [True, False, False]])
result = grid2graph(grid)

graph_plot(result)

print('\nTESTS:')

closed = [[1], [2]]
expandListTest = graph_getExpandList(graphVectorMedium, 3, closed)
print(expandListTest)

end = graph_search(graphVectorMedium,0,8)
print(end[0])

def createEnvironment():
  tfGrid = numpy.full((30,30), True, dtype=bool)
  for i in range(10, 20):
    for j in range(18, 20):
      tfGrid[i][j] = False

  for i in range(18, 20):
    for j in range(10, 20):
      tfGrid[i][j] = False

  thirtyArray = []
  for i in range(len(tfGrid[0])):
    thirtyArray.append(i)

  thirtyGrid = grids(thirtyArray,  thirtyArray, tfGrid)
  thirtyGraph = grid2graph(thirtyGrid)
  return thirtyGraph

thirtyGraph = createEnvironment()

matPlotGraph(thirtyGraph)
plt.pause(10)
plt.clf();
plt.pause(10)
#matPlotGraph(graphVectorMedium)
animated_graph_search(thirtyGraph, 35, 829)

print("End of Program")
