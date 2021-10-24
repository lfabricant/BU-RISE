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
  print(grid.xx)
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
  #print(idxStart)
  #print(idxGoal)
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

#def updateGraphFinal(xpoints, ypoints, shapes, colours, maxX, maxY, graphVector, xPath):
def updateGraphFinal(xPath):
  #print(xPath)
  """"
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
  """
  for i in range(len(xPath)):
    plt.scatter(xPath[i][0], xPath[i][1], c = 'green')
  #marker = shapes[i], zorder = 2

  for i in range(len(xPath)-1):
    plt.plot(numpy.array([xPath[i][0], xPath[i+1][0]]), numpy.array([xPath[i][1], xPath[i+1][1]]), c = 'green')
  plt.show()
  """"
  plt.show(block = True)
  """


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
  
def createEnvironment(l, radius, coords, obstacle=None, side=None):
  tfGrid = numpy.full((l,l), False, dtype=bool)
  for c in coords:
    print('x y points')
    print(int(c[0].round()))
    print(int(c[1].round()))
    print()

    #round((obstacle[3]-obstacle[2])/2+obstacle[2])

    #print('HERE: ', c[0], ' ', c[1])
    #print('ROUND: ', int(c[0].round()), ' ', int(c[1].round()))
    
    tfGrid[int(c[0].round())][int(c[1].round())] = True
    if(int(c[0].round()) < (c[0])):
      tfGrid[int(c[0].round())+1][int(c[1].round())] = True
    if(int(c[1].round()) < (c[1])):
      tfGrid[int(c[0].round())][int(c[1].round())+1] = True
    #tfGrid[int(c[0].round())+1][int(c[1].round())+1] = True
    if int(c[0].round()) > 0 and int(c[1].round()) > 0: 
      if(int(c[0].round()) > (c[0])):
        tfGrid[int(c[0].round())-1][int(c[1].round())] = True
      if(int(c[1].round()) > (c[1])):
        tfGrid[int(c[0].round())][int(c[1].round())-1] = True
      #tfGrid[int(c[0].round())-1][int(c[1].round())-1] = True
      #tfGrid[int(c[0].round())-1][int(c[1].round())+1] = True
      #tfGrid[int(c[0].round())+1][int(c[1].round())-1] = True

  

  
  if obstacle:
    x = numpy.linspace(obstacle[0], obstacle[1], obstacle[1]-obstacle[0]+1)
    y = numpy.linspace(obstacle[2], obstacle[3], obstacle[3]-obstacle[2]+1)
    for i in x:
      for j in y:
        if side == 1 and (i != obstacle[0] or j != round((obstacle[3]-obstacle[2])/2+obstacle[2])):
          tfGrid[int(i)][int(j)] = False
        if side == 2 and (i != obstacle[1] or j!= round((obstacle[3]-obstacle[2])/2+obstacle[2])):
          tfGrid[int(i)][int(j)] = False

  for i in tfGrid:
    print(i)
    
  thirtyArray = []
  for i in range(len(tfGrid[0])):
    thirtyArray.append(i)
    print(thirtyArray)

  thirtyGrid = grids(thirtyArray,  thirtyArray, tfGrid)
  thirtyGraph = grid2graph(thirtyGrid)

  for i in range(len(thirtyGraph)):
      if thirtyGraph[i].x[0][0] == int(coords[0][0].round()) and thirtyGraph[i].x[1][0] == int(coords[0][1].round()):
        startIdx = i
  if obstacle:
    if side == 1: #LEFT
      eX = obstacle[0]
      eY = round((obstacle[3]-obstacle[2])/2+obstacle[2])
      #print(eX, ' ', eY)
    else: #RIGHT
      eX = obstacle[1]
      eY = round((obstacle[3]-obstacle[2])/2+obstacle[2])
      #print(eX, ' ', eY)
    for i in range(len(thirtyGraph)):
      if thirtyGraph[i].x[0][0] == eX and thirtyGraph[i].x[1][0] == eY:
        endIdx = i
  else:
    for i in range(len(thirtyGraph)):
      #print(int(coords[0][0].round()), ' ', int(coords[0][1].round()), ' : ', thirtyGraph[i].x)
      if thirtyGraph[i].x[0][0] == int(coords[len(coords)-1][0].round()) and thirtyGraph[i].x[1][0] == int(coords[len(coords)-1][1].round()):
        endIdx = i
  #endIdx = 0
  return [thirtyGraph, startIdx, endIdx] 

def addObstacle(start, end, l, radius, coords):
  tfGrid = numpy.full((l,l), False, dtype=bool)
  for c in coords:
    #print('HERE: ', c[0], ' ', c[1])
    #print('ROUND: ', int(c[0].round()), ' ', int(c[1].round()))
    if start > int(c[0].round()) and end < int(c[0].round()):
      tfGrid[int(c[0].round())][int(c[1].round())] = True
      if(int(c[0].round()) < (c[0])):
        tfGrid[int(c[0].round())+1][int(c[1].round())] = True
      if(int(c[1].round()) < (c[1])):
        tfGrid[int(c[0].round())][int(c[1].round())+1] = True
      #tfGrid[int(c[0].round())+1][int(c[1].round())+1] = True
      if int(c[0].round()) > 0 and int(c[1].round()) > 0: 
        if(int(c[0].round()) > (c[0])):
          tfGrid[int(c[0].round())-1][int(c[1].round())] = True
        if(int(c[1].round()) > (c[1])):
          tfGrid[int(c[0].round())][int(c[1].round())-1] = True
      #tfGrid[int(c[0].round())-1][int(c[1].round())-1] = True
      #tfGrid[int(c[0].round())-1][int(c[1].round())+1] = True
      #tfGrid[int(c[0].round())+1][int(c[1].round())-1] = True
  
  thirtyArray = []
  for i in range(len(tfGrid[0])):
    thirtyArray.append(i)

  thirtyGrid = grids(thirtyArray,  thirtyArray, tfGrid)
  thirtyGraph = grid2graph(thirtyGrid)

  """"
  for i in range(len(thirtyGraph)):
    print(int(coords[0][0].round()), ' ', int(coords[0][1].round()), ' : ', thirtyGraph[i].x)
    if thirtyGraph[i].x[0][0] == int(coords[0][0].round()) and thirtyGraph[i].x[1][0] == int(coords[0][1].round()):
      startIdx = i
    
    if thirtyGraph[i].x[0][0] == int(coords[len(coords)-1][0].round()) and thirtyGraph[i].x[1][0] == int(coords[len(coords)-1][1].round()):
      endIdx = i
  """
  return [thirtyGraph, start, end] 


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
        plt.plot(numpy.array([node.x[0], vector[neighbor[0]-1].x[0]]), numpy.array([node.x[1], vector[neighbor[0]-1].x[1]]), 'k')
    count += 1

  plt.ion()
  plt.scatter(xpoints, ypoints, color='black')
  #plt.xlim([0, maxX])
  #plt.ylim([0, maxY])
  
  plt.show()

