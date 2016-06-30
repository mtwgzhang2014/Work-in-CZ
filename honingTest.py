#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# test case

from simLC import *
import numpy as np
from datetime import datetime
from scipy.optimize import fsolve

# preset #################################################
# init_Left = 558028;
# init_Right = 555806; 
# init_Up = 558708;

# L:-1975 T:-2288
# init_Left = 645528 - 166394;
# init_Right = 644170 - 166394; 
# init_Up = 644873 - 166394;

# L:
# init_Left = 648955 - 165638;
# init_Right = 647852 - 165638; 
# init_Up = 648804 - 165638

# L:6483 T:3817
# init_Left = 620805 - 170220;
# init_Right = 622531 - 170220; 
# init_Up = 619599 - 170220;

# L:-313 T:5125
init_Left = 866181 - 87035;
init_Right = 870195 - 87035; 
init_Up = 870441 - 87035;




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
tmpData = np.zeros(2, dtype=np.int64)

def simpleFit(alphaL, alphaT, LongError, TranError, order):
	for m in range(2):
		for n in range(2):
			if ((-1)**n*alphaL-(-1)**m*alphaL/alphaT*(-1)**n*alphaT) != 0:
				try:
					tmpData[0] = (LongError - (-1)**m*alphaL/alphaT*TranError)/\
					((-1)**n*alphaL-(-1)**m*alphaL/alphaT*(-1)**n*alphaT)
				except Exception as e:
					print(e)
					print("1",(-1)**n*alphaL-(-1)**m*alphaL/alphaT*(-1)**n*alphaT)
					print("2",LongError - (-1)**m*alphaL/alphaT*TranError)
					print(tmpData[0])
			else:
				tmpData[0] = negInf
			#print("m, n : %d, %d -- %f" % (m,n,tmpData[0]))
			# select POS1 or POS2 first
			if tmpData[0] > 0:
				curHonePos[0] = n + 1 # map the sign parameter n to POS
				curHoneStep[0] = np.round(tmpData[0]**(1.0/order)*0.9)
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
						curHoneStep[1] = np.round(tmpData[1]**(1.0/order)*0.9)

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
#print(alphaL,alphaT)

simpleFit(alphaL, alphaT, Long[0], Tran[0], 3)
testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))

#####
# if curPOS == prePOS: 
# 	Long[cur]-Long[cur-1]
# else:
#	Long[cur]-Long[0] or Long[cur]-Long[lastFind]

alphaL = (Long[2] - Long[0])/(sign(curHonePos[0],"Long")*curHoneStep[0]**3 + sign(curHonePos[1],"Long")*curHoneStep[1]**3)
alphaT = (Tran[2] - Tran[0])/(sign(curHonePos[0],"Tran")*curHoneStep[0]**3 + sign(curHonePos[1],"Tran")*curHoneStep[1]**3)
#print(alphaL,alphaT)
simpleFit(alphaL, alphaT, Long[0], Tran[0], 3)
testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))



# alphaL = (Long[3] - Long[0])/(sign(curHonePos[0],"Long")*curHoneStep[0]**3 + sign(curHonePos[1],"Long")*curHoneStep[1]**3)
# alphaT = (Tran[3] - Tran[0])/(sign(curHonePos[0],"Tran")*curHoneStep[0]**3 + sign(curHonePos[1],"Tran")*curHoneStep[1]**3)
# #print(alphaL,alphaT)

# simpleFit(alphaL, alphaT, Long[0], Tran[0], 3)
# testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
# testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
# print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
# Long.append(testLC.LongError)
# Tran.append(testLC.TranError)
# print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))



# alphaL = (Long[4] - Long[0])/(sign(curHonePos[0],"Long")*curHoneStep[0]**3 + sign(curHonePos[1],"Long")*curHoneStep[1]**3)
# alphaT = (Tran[4] - Tran[0])/(sign(curHonePos[0],"Tran")*curHoneStep[0]**3 + sign(curHonePos[1],"Tran")*curHoneStep[1]**3)
# #print(alphaL,alphaT)

# simpleFit(alphaL, alphaT, Long[0], Tran[0], 3)
# testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
# testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
# print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
# Long.append(testLC.LongError)
# Tran.append(testLC.TranError)
# print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))

# testLC.loadCellOutput(1500,1)
# testLC.loadCellOutput(1700,0)

testLC.loadCellOutput(1020,1)
testLC.loadCellOutput(1600,3)
print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
Long.append(testLC.LongError)
Tran.append(testLC.TranError)
print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))

#############################################################################################################
# new test

#print(testLC.honeStepRecord)
# honeStep4Tran = np.zeros([2,testLC.honeStepRecord.shape[0],4])
# honeStep4Long = np.zeros([2,testLC.honeStepRecord.shape[0],4])

# for i in range(testLC.honeStepRecord.shape[0]):
# 	for j in range(4):
# 		honeStep4Tran[0,i,j] = sign(j,"Tran")*testLC.honeStepRecord[i,j]**3
# 		honeStep4Tran[1,i,j] = sign(j,"Tran")*testLC.honeStepRecord[i,j]**2
# 		honeStep4Long[0,i,j] = sign(j,"Long")*testLC.honeStepRecord[i,j]**3
# 		honeStep4Long[1,i,j] = sign(j,"Long")*testLC.honeStepRecord[i,j]**2


# alphaT = np.linalg.lstsq(sum(honeStep4Tran.transpose()), np.frombuffer(Tran[1:]-Tran[0], dtype=float))[0]
# alphaL = np.linalg.lstsq(sum(honeStep4Long.transpose()), np.frombuffer(Long[1:]-Long[0], dtype=float))[0]
# #print(honeStep4Tran)
# print(alphaT)
# print(alphaL)

# honePos = curHonePos

# def fModel(step):
# 	s1 = step[0]
# 	s2 = step[1]
# 	calTran = alphaT[0]*(sign(honePos[0],"Tran")*s1**3+sign(honePos[1],"Tran")*s2**3) + alphaT[1]*(sign(honePos[0],"Tran")*s1**2+sign(honePos[1],"Tran")*s2**2) + Tran[0]
# 	calLong = alphaL[0]*(sign(honePos[0],"Long")*s1**3+sign(honePos[1],"Long")*s2**3) + alphaL[1]*(sign(honePos[0],"Long")*s1**2+sign(honePos[1],"Long")*s2**2) + Long[0]
# 	return [calTran, calLong]

# curHoneStep = np.round(fsolve(fModel, curHoneStep))
# print("result is: ", curHoneStep)

# testLC.loadCellOutput(curHoneStep[0],curHonePos[0])
# testLC.loadCellOutput(curHoneStep[1],curHonePos[1])
# print("Honing: %d POS%d; %d POS%d" % (curHoneStep[0],curHonePos[0],curHoneStep[1],curHonePos[1]))
# Long.append(testLC.LongError)
# Tran.append(testLC.TranError)
# print("Long Error: %d; Tran Error: %d " % (Long[-1],Tran[-1]))