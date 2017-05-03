import numpy as np
import sys, time, os
import h5py as h5
import matplotlib.pyplot as plt
#Add Modules from other directories
developerDirectory = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDirectory = developerDirectory + "tools/"
sys.path.append( toolsDirectory )


nParticles = 32**3
totalSteps = 1000


G    = 1 #m**2/(kg*s**2)
mSun = 1     #kg
pMass = 1
initialR =  500


# #Initialize file for saving data
# outputDir = '/home/bruno/data/nBody/'
# # if not os.path.exists( outputDir ): os.makedirs( outputDir )
# outDataFile = outputDir + 'data_all.h5'
# dFile = h5.File( outDataFile , "w")
# posHD = dFile.create_group("pos")

#Domain Parameters
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
z_min, z_max = 0.0, 1.0

#Particles Positions
p_pos_x = (x_max - x_min)*np.random.random( nParticles ) + x_min
p_pos_y = (y_max - y_min)*np.random.random( nParticles ) + y_min
p_pos_z = (z_max - z_min)*np.random.random( nParticles ) + z_min

#Particles velocities
p_vel_x = np.zeros( nParticles )
p_vel_y = np.zeros( nParticles )
p_vel_z = np.zeros( nParticles )

#Grid Properties
nPoints = 128
nCells_x = nPoints
nCells_y = nPoints
nCells_z = nPoints
nCells = nCells_x*nCells_y*nCells_z

dx = ( x_max - x_min ) /  nCells_x
dy = ( y_max - y_min ) /  nCells_y
dz = ( z_max - z_min ) /  nCells_z

#Density array
rho = np.zeros(nCells)

#Nearest Grid points
idx_x = ( p_pos_x / dx ).astype(np.int)
idx_y = ( p_pos_y / dy ).astype(np.int)
idx_z = ( p_pos_z / dz ).astype(np.int)
for i in range(nParticles):
  idx = idx_x[i] + idx_y[i]*nCells_x + idx_z[i]*nCells_x*nCells_y
  # print idx
  rho[idx] += 1  #assuming all particles have equal mass
# rho = rho.reshape([nCells_z, nCells_y, nCells_x])
# rho *= dx*dy*dz
