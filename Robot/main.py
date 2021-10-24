#!/usr/bin/env python
import numpy as np 
import matplotlib.pyplot as plt 
import AStar


class Node(object):
  def __init__(self, angle, points, coords):
    self.angle = angle
    self.points = points
    self.coords = coords

def sind(x):
    return np.sin(x * np.pi / 180)

def cosd(x):
    return np.cos(x * np.pi / 180)

def createArm(k, angle, radius, size):
  if k == 0:
    #Create Left Arm
    xL = np.linspace(0,-np.pi*radius,180) 
    yL = [radius*2]*180
    armL = [None]*180
    for i in range(len(armL)):
        x = xL[i] + size
        y = yL[i] 
        a = 0
        armL[i] =  Node(a, 0, [x,y])
    #Create Right Arm
    xR = np.linspace(0,np.pi*radius,180)
    yR = [radius*2]*180
    armR = [None]*180
    for i in range(len(armR)):
        x = xR[i] + size
        y = yR[i]
        a = 0
        armR[i] =  Node(a, 0, [x,y])
  else:  
    #Create Left Arm                   
    armL = [None]*len(angle)
    #Create Right Arm
    armR = [None]*len(angle)
    for i in range(len(angle)):
      #print(angle[i])
      x = (180/k) * ( sind( angle[i] ) * radius) 
      y = -(180/k) * ( radius + cosd( angle[i] ) * radius) + radius*2
      #Add Coordinates, Right Arm has opposite x-values
      armL[i] =  Node(angle[i], 0, [-x+size,y])
      armR[i] =  Node(angle[i], 0, [x+size,y])
  return [armL, armR]

def plotArm(arm, l, radius):
  armL = arm[0]
  armR = arm[1]

  plt.clf()
  plt.xlim([0, l-1])
  plt.ylim([-radius/2, (l-1)/2])  #0
  plt.ion()

  for i in range(len(armL)-1):
    #Plot right arm
    plt.plot(np.array([armL[i].coords[0], armL[i+1].coords[0]]), np.array([armL[i].coords[1], armL[i+1].coords[1]]), 'y', zorder = 1) 

    #Plot left arm
    plt.plot(np.array([armR[i].coords[0], armR[i+1].coords[0]]), np.array([armR[i].coords[1], armR[i+1].coords[1]]), 'b', zorder = 1)

  #Plot dots to show start and stop
  plt.plot(armR[0].coords[0], armR[0].coords[1],'go', markersize = 5)
  plt.plot(armL[len(armL)-1].coords[0],armL[len(armL)-1].coords[1],'ro', markersize = 5)
  plt.plot(armR[len(armR)-1].coords[0],armR[len(armR)-1].coords[1],'ro', markersize = 5)

  plt.show()
  plt.pause(0.0000001)


def setAngle(i, s, f):
  #Force is curr
  angleInitial = 0
  angleFinal = i*(180/s) # aka k
  angle = np.linspace(angleInitial+180,180-angleFinal,s)
  #print(angle)
  if f > 0:
    for i in range(int(len(angle)/2)):
      angle[i+1] = angle[i+1]-f/100
      angle[len(angle)-1-i] = angle[len(angle)-1-i]+f/100
    """"
    for i in range(len(angle)-1):
      angle[i+1] = angle[i+1]-f/10
    """
  elif f < 0:
    for i in range(int(len(angle)/2)):
      angle[i+1] = angle[i+1]-f/100
      angle[len(angle)-1-i] = angle[len(angle)-1-i]+f/100
    """"
    for i in range(len(angle)-1):
      angle[i+1] = angle[i+1]-f/10
    """
  #print(angle)
  return [angleFinal, angle]

def simulate(size, radius, l, force):
  plt.xlim([0, l-1])
  plt.ylim([-radius/2, (l-1)/2])
  for i in range(size+1):
      #print(i)
      a = setAngle(i, size, force)
      arm = createArm(a[0], a[1], radius, size)
      plt.pause(1) # slow down start
      plotArm(arm, l, radius)

def plotFinalOnly(size, radius, l, force):
  a = setAngle(size, size, force)
  arm = createArm(a[0], a[1], radius, size)
  plotArm(arm, l, radius)
  #print(arm[0][len(arm[0])-1].coords[0],arm[0][len(arm[0])-1].coords[1])

def plotObstacle(obstacle):
  #startX = 9
  #endX = 11
  #startY = 0
  #endY = 2

  #plt.plot(startX,armL[len(armL)-1].coords[1],'ro', markersize = 5)
  plt.ion()
  x = np.linspace(obstacle[0], obstacle[1], obstacle[1]-obstacle[0]+1)
  y = np.linspace(obstacle[2], obstacle[3], obstacle[3]-obstacle[2]+1)
  for i in x:
    for j in y:
      #print(i, ' ' , j)
      plt.plot(i,j,'mx', markersize = 5)


def start():
  #move obstacle down
  size = 40 # 180, nodes = angles
  radius = size/np.pi
  l = size*2+1#-int(radius)+1

  #size = 9 # 180, nodes = angles
  #radius = l/np.pi
  #l = 5

  #l = radius*2
  force = 0
  #force, radius, l = int(input('Enter a value: '))

  #plt.xlim([0, 10])
  #plt.ylim([0, 10]) 
 
  if force > 50 or force < -50:
    print('Too Much Force')
    #return None #can comment it out
  
  #TESTS
  plt.pause(10)
  #simulate(size, radius, l, force)
  #plotFinalOnly(size, radius, l, force)
  plt.pause(10)



  #ASTAR - soon switch false and trues of arm
  # """"
  a = setAngle(size, size, force)
  arm = createArm(a[0], a[1], radius, size)
  coords1 = [None]*len(arm[0])
  for p in range(len(arm[0])):
    pX = arm[0][p].coords[0]
    pY = arm[0][p].coords[1]
    coords1[p] = [pX,pY]

  #print('coordsX1')
  #print(coords1)


  coords2 = [None]*len(arm[1])
  for p in range(len(arm[1])):
    pX = arm[1][p].coords[0]
    pY = arm[1][p].coords[1]
    coords2[p] = [pX,pY]
  #coords = coords1 + coords2
  
  
  
  #"""
  #LEFT ARM
  e = AStar.createEnvironment(l, radius, coords1)
  #print(e)
  plt.ion()
  AStar.matPlotGraph(e[0]) 
  #print(e[1])
  #print(e[2])
  #print(AStar.graph_search(e[0], e[1], e[2])[0]) 
  
  #plt.ion()
  #RIGHT ARM
  e = AStar.createEnvironment(l, radius, coords2)
  #print(e)
  plt.ion()
  AStar.matPlotGraph(e[0]) 
  #print(AStar.graph_search(e[0], e[1], e[2])[0]) 

  plt.pause(10)
  plt.clf()
  
  #print('Here 1')
  #""" ASTAR
  startX = size-1
  endX = size+1
  startY = 0
  endY = 2
  obstacle = [startX, endX, startY, endY]
  #print('Here 2')
  plotObstacle(obstacle)
  #"""
  #LEFT ARM
  e1 = AStar.createEnvironment(l, radius, coords1, obstacle, 1)
  plt.ion()
  AStar.matPlotGraph(e1[0]) 
  #AStar.updateGraphFinal(AStar.graph_search(e1[0], e1[1], e1[2])[0])
  #print(AStar.graph_search(e1[0], e1[1], e1[2])[0])

  #RIGHT ARM
  e2 = AStar.createEnvironment(l, radius, coords2, obstacle, 2)
  plt.ion()
  AStar.matPlotGraph(e2[0]) 
  #AStar.updateGraphFinal(AStar.graph_search(e2[0], e2[1], e2[2])[0])
  #print(AStar.graph_search(e2[0], e2[1], e2[2])[0])

  plt.pause(10)

  plt.clf()
  plotObstacle(obstacle)
  AStar.updateGraphFinal(AStar.graph_search(e1[0], e1[1], e1[2])[0])
  AStar.updateGraphFinal(AStar.graph_search(e2[0], e2[1], e2[2])[0])

  plt.pause(20)
  #"""

 # plt.pause(5)
  #obstacleGrid = np.full((2,10), False, dtype=bool)

  #AStar.addObstacle(9, 10, l, radius, coords)
  
  #print(AStar.graph_search(e[0], e[1], e[2])[0]) 
  #a = setAngle(size, size, force)
  #arm = createArm(a[0], a[1], radius)
  #for i in arm

start()

