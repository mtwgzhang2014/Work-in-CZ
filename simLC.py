#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# load cell simulation

import numpy as np 
from array import array
from enum import Enum,unique

@unique
class HonePos(Enum):
	Up = 0
	Other = 1
	Right = 2
	Left = 3

class simLC:
	def __init__(self,initUp,initRight,initLeft,noiseSigma):
		self.Up = initUp
		self.Right = initRight
		self.Left = initLeft
		self.Other = initLeft+initUp-initRight
		self.initLongError = np.round((self.Right - self.Up)/self.Right*1e6)
		self.initTranError = np.round((self.Right - self.Left)/self.Right*1e6)
		self.Noise = np.round(noiseSigma * np.random.randn(4,1))
		self.cornerTheta = np.zeros((4,4))
		self._tmpRecord = array("i")
		self.adjRecord = np.zeros((1,4))
		self._tmpStepRecord = array("i")
		self._honeStepRecord = np.array([])
		self._tmpPosRecord = array("i")
		self._honePosRecord = np.array([])

	@property
	def LongError(self):
	    return np.round((self.Right - self.Up)/self.Right*1e6)
	@property
	def TranError(self):
	    return np.round((self.Right - self.Left)/self.Right*1e6)

	@property
	def honeStepRecord(self):
		tmpSize = self._honeStepRecord.size
		if  tmpSize % 2 == 0:
			tmpHoneStep = np.zeros([tmpSize/2,4])
			#print(tmpHoneStep)
			for i in range(int(tmpSize/2)):
				for j in range(2):
					tmpHoneStep[i][self._honePosRecord[i*2+j]] = self._honeStepRecord[i*2+j]
		else:
			tmpHoneStep = np.zeros([int(tmpSize/2)+1,4])
			#print(tmpHoneStep)
			for i in range(int(tmpSize/2)):
				for j in range(2):
					tmpHoneStep[i][self._honePosRecord[i*2+j]] = self._honeStepRecord[i*2+j]
			tmpHoneStep[-1][self._honePosRecord[-1]] = self._honeStepRecord[-1]
		return tmpHoneStep
	

	def _calcHoneAdj(self,honeStep,honePos):
		#np.seterr(all = 'warn')
		if honePos == 0: # POS Up
			self.cornerTheta[0,:] = np.array([2.0572e-7, 	0.001046,	0,	80])	# Up corner parameter			
			self.cornerTheta[2,:] = np.array([1.4275e-7,	0.0007437,	0,	54])	# Right corner parameter
			self.cornerTheta[3,:] = np.array([6.3343e-8, 	-4.3617e-5,	0,	37])	# Left corner parameter
		elif honePos == 1: # POS Other
			self.cornerTheta[0,:] = np.array([9.7984e-8,	0.0009047,	0,	-54])
			self.cornerTheta[2,:] = np.array([-3.9565e-8,	0.0003041,	0,	-1])
			self.cornerTheta[3,:] = np.array([3.0545e-7,	0.001797,	0,	-14])
		elif honePos == 2: # POS Right
			self.cornerTheta[0,:] = np.array([-1.1985e-8,	0.0008786,	0,	311])
			self.cornerTheta[2,:] = np.array([5.1397e-8,	0.001171,	0,	355])
			self.cornerTheta[3,:] = np.array([2.8902e-8,	0.0001329,	0,	306])			
		elif honePos == 3: # POS Left
			self.cornerTheta[0,:] = np.array([-6.1081e-8,	-6.3095e-5,	0,	27])
			self.cornerTheta[2,:] = np.array([1.3651e-8,	0.0001839,	0,	59])
			self.cornerTheta[3,:] = np.array([1.3890E-7,	0.0008259,	0,	2])
		else:
			raise ValueError("honePos value %d invalid!", honePos)

		honeStep = np.int64(honeStep)
		stepVector = np.array([pow(honeStep,3),pow(honeStep,2),honeStep,1])	
		#print(type(stepVector[0]))
		adjRight = np.round(np.dot(stepVector,self.cornerTheta[2,:].transpose())+self.Noise[2])
		adjLeft = np.round(np.dot(stepVector,self.cornerTheta[3,:])+self.Noise[3])
		adjUp = np.round(np.dot(stepVector,self.cornerTheta[0,:])+self.Noise[0])
		adjOther = np.round(adjLeft + adjUp - adjRight + self.Noise[1])
		#print(self.adjUp,self.adjOther,self.adjRight,self.adjLeft)
		self._tmpRecord.append(adjUp)
		self._tmpRecord.append(adjOther)
		self._tmpRecord.append(adjRight)
		self._tmpRecord.append(adjLeft)
		self.adjRecord = np.frombuffer(self._tmpRecord, dtype=np.int).reshape(-1,4)
		#print(self.adjRecord)
		self._tmpStepRecord.append(honeStep) # must be int
		self._honeStepRecord = np.frombuffer(self._tmpStepRecord,dtype=np.int)
		self._tmpPosRecord.append(honePos) # must be int
		self._honePosRecord = np.frombuffer(self._tmpPosRecord,dtype=np.int)
		return np.array([adjUp, adjOther, adjRight, adjLeft])

	def loadCellOutput(self,honeStep,honePos):
		if not honePos in self._honePosRecord:
			adjArray = self._calcHoneAdj(honeStep,honePos)
			self.Up 	+= 	adjArray[HonePos.Up.value]
			self.Other 	+= 	adjArray[HonePos.Other.value]
			self.Right 	+= 	adjArray[HonePos.Right.value]
			self.Left 	+= 	adjArray[HonePos.Left.value]
		else:
			prevHP_i =  np.where(self._honePosRecord == honePos)[0][-1] # previous HonePos Index in adjRecord
			adjArray = self._calcHoneAdj(honeStep,honePos)
			self.Up 	= self.Up - self.adjRecord[prevHP_i, HonePos.Up.value] + adjArray[HonePos.Up.value]
			self.Other 	= self.Other - self.adjRecord[prevHP_i, HonePos.Other.value] + adjArray[HonePos.Other.value]
			self.Right 	= self.Right - self.adjRecord[prevHP_i, HonePos.Right.value] + adjArray[HonePos.Right.value]
			self.Left 	= self.Left - self.adjRecord[prevHP_i, HonePos.Left.value] + adjArray[HonePos.Left.value]
		return np.array([self.Up, self.Other, self.Right, self.Left])

if __name__ == "__main__":
	testLC = simLC(558708, 555806, 558028, 10)
	print("The initial Long error is %d and the Tran error is %d " % (testLC.initLongError,testLC.initTranError))
	#print(testLC.calcHoneAdj(1655,2))
	#print(testLC.calcHoneAdj(860,3))\
	# first honing
	testLC.loadCellOutput(1655,2)
	testLC.loadCellOutput(860,3)
	# second honing
	testLC.loadCellOutput(1919,2)
	testLC.loadCellOutput(1532,3)
	# third honing
	testLC.loadCellOutput(2050,2)
	testLC.loadCellOutput(1705,3)

	testLC.loadCellOutput(190,0)
	print(testLC.adjRecord)
	print(testLC.honeStepRecord)
	print("The Long error is %d and the Tran error is %d " % (testLC.LongError,testLC.TranError))



