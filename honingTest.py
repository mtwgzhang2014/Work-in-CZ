#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# test case

from simLC import *
import numpy as np
from datetime import datetime

# preset #################################################
# init_Left = 558028;
# init_Right = 555806; 
# init_Up = 558708;

init_Left = 649717 - 166812;
init_Right = 648354 - 166812; 
init_Up = 649243 - 166812;

Noise_Sigma = 20

#honeStep = np.zeros(1, 4);
curHoneStep = np.zeros(2, dtype=np.int64);
curHonePos = np.zeros(2, dtype=int);

Long = []
Tran = []

negInf = -99999999999999

#np.seterr(over = "ignore", divide = "ignore") # ignore floating-point errors are handles
##########################################################

print("Auto-Honing Experiment")
now = datetime.now()
print(now.strftime('%Y-%m-%d %H:%M:%S'))
testLC = simLC(init_Up, init_Right, init_Left, Noise_Sigma)
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("The initial Long error is %d and the Tran error is %d " % (Long[-1],Tran[-1]))

##########################################################
# parameter by heuristic methods, best one is 0.0015 & 0.002 
alphaL = 0.003
alphaT = 0.005
##########################################################

# first honing ###########################################
tmpData = np.zeros(2)

def simpleFit(alphaL, alphaT, LongError,TranError,order):
	for m in range(2):
		for n in range(2):
			if ((-1)**n*alphaL-(-1)**m*alphaL/alphaT*(-1)**n*alphaT) != 0:
				tmpData[0] = (LongError - (-1)**m*alphaL/alphaT*TranError)/\
				((-1)**n*alphaL-(-1)**m*alphaL/alphaT*(-1)**n*alphaT)
			else:
				tmpData[0] = negInf
			#print("m, n : %d, %d -- %f" % (m,n,tmpData[0]))
			# select POS1 or POS2 first
			if tmpData[0] > 0:
				curHonePos[0] = n + 1 # map the sign parameter n to POS
				curHoneStep[0] = np.round(tmpData[0]**(1.0/order))
				# then POS0 or POS3
				for k in range(2):
					tmpData[1] = (TranError-(-1)**n*alphaT*curHoneStep[0]**order)/\
					((-1)**k*alphaT)
					#print("k, n : %d, %d -- %f" % (k,n,tmpData[1]))
					if tmpData[1] > 0:
						if k==0:
							curHonePos[1] = 3
						else:
							curHonePos[1] = 0
						curHoneStep[1] = np.round(tmpData[1]**(1.0/order))

simpleFit(alphaL, alphaT, Long[0], Tran[0], 2)
testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))

def sign(POS,errorDir):
	if errorDir == "Long":
		if POS == 0 or POS == 1:
			return -1
		elif POS == 2 or POS == 3:
			return 1
		else:
			raise ValueError("POS is wrong!")
	elif errorDir == "Tran":
		if POS == 1 or POS == 3:
			return -1
		elif POS == 0 or POS == 2:
			return 1
		else:
			raise ValueError("POS is wrong!")
	else:
		raise ValueError("errorDir is wrong!")


alphaL = (Long[1] - Long[0])/(sign(curHonePos[0],"Long")*curHoneStep[0]**3 + sign(curHonePos[1],"Long")*curHoneStep[1]**3)
alphaT = (Tran[1] - Tran[0])/(sign(curHonePos[0],"Tran")*curHoneStep[0]**3 + sign(curHonePos[1],"Tran")*curHoneStep[1]**3)
print(alphaL,alphaT)

simpleFit(alphaL, alphaT, Long[0], Tran[0], 3)
testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))


#simpleFit(testLC.LongError,testLC.TranError,3)


#testLC.loadCellOutput(1344,2)
#testLC.loadCellOutput(898,3)
#print("")
##########################################################
# second honing
# testLC.loadCellOutput(1919,2)
# testLC.loadCellOutput(1532,3)
# # third honing
# testLC.loadCellOutput(2050,2)
# testLC.loadCellOutput(1705,3)

# testLC.loadCellOutput(190,0)
#print(testLC.adjRecord)
