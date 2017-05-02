import numpy as np
import sys, time, os
import h5py as h5
import matplotlib.pyplot as plt
#Add Modules from other directories
developerDirectory = '/home/bruno/Desktop/Dropbox/Developer/'
toolsDirectory = developerDirectory + "tools/"
sys.path.append( toolsDirectory )


nParticles = 1024*64
totalSteps = 1000
