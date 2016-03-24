#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

__author__ = 'Mason Zhang'

# Name: xmlFilter.py 
# Purpose: 解析XML文件，提取并处理角差记录数据
#  1. 提取XML中的特定数据，DATA中数据部分内容
#  2. 将DATA中第5行到第12行数据放入一个二维数组中
#  3. 记录第一行的初始角差值 
#  4. 最终计算出每个XML文件中对各角的最终进刀量

import re

# filePath = r'C:\Users\Administrator\Desktop\7283523086-6-20150106124217-MeasData.xml'


def obtainKeyData(filePath):
	# 关键数据 L_error, T_error, up_pos, other_pos, right_pos, left_pos
	keyData = [0]*6
	style12Flag = False
	# 获取纯角差数据
	pattern16 = re.compile("X4(.*?)EndTime:", re.S)
	pattern12= re.compile("Hone_Depth2(.*?)EndTime:", re.S) 
	# re.S 使 . 匹配包括换行在内的所有字符
	# 在此可以避免换行符\n不能被匹配

	with open(filePath, 'r') as xmlfile:
		tmpXml = xmlfile.read()
		# re.search 扫描整个字符串并返回第一个成功的匹配
		# 而re.match 尝试从字符串的起始位置匹配一个模式，
		# 如果不是起始位置匹配成功的话，match()就返回none
		tmpData = re.search(pattern16, tmpXml)
		if tmpData == None:
			tmpData = re.search(pattern12, tmpXml)
			style12Flag = True

	if tmpData!=None:
		dataList = tmpData.group(1).split()
		g = iter(dataList)
		if len(dataList) % 12 == 0 and float(dataList[15]) > 1 and style12Flag:
			rowNum = int(len(dataList)/12)
			# 下面是两种将一维List转变为二维List的语句
			# 1. 采用迭代器结果逐个赋值到二维List中，[[0 for..] for..]二维数组
			# 2. 利用zip函数的特性，但结果为List中包含tuple的形式
			dataArray = [[next(g) for _ in range(12)] for _ in range(rowNum)]
			# * unpack the arguments out of a list or tuple
			#dataArray = list(zip(*[iter(dataList)]*12))
		elif len(dataList) % 16 == 0 and float(dataList[15]) < 1 and (not style12Flag):
			rowNum = int(len(dataList)/16)
			dataArray = [[next(g) for _ in range(16)] for _ in range(rowNum)]
		else:
			return 'Bad data!'
		# L_error
		keyData[0] = int(dataArray[0][4])
		# T_error
		keyData[1] = int(dataArray[0][5])

		if style12Flag:
			for i in range(len(dataArray)):
				if dataArray[i][6] == '0':
					keyData[2] += int(dataArray[i][8])
				elif dataArray[i][6] == '1':
					keyData[3] += int(dataArray[i][8])
				elif dataArray[i][6] == '2':
					keyData[4] += int(dataArray[i][8])
				elif dataArray[i][6] == '3':
					keyData[5] += int(dataArray[i][8])
				else:
					raise ValueError('E01: The original xml has some mistake!')
				if dataArray[i][9] == '0':
					keyData[2] += int(dataArray[i][11])
				elif dataArray[i][9] == '1':
					keyData[3] += int(dataArray[i][11])
				elif dataArray[i][9] == '2':
					keyData[4] += int(dataArray[i][11])
				elif dataArray[i][9] == '3':
					keyData[5] += int(dataArray[i][11])
				else:
					raise ValueError('E02: The original xml has some mistake!')
				return 'Bad data!'
		else:
			#lastRow = len(dataArray) - 1
			keyData[2] = round(float(dataArray[-1][-4]) / 0.000125) # C0 for up -- X1
			keyData[3] = round(float(dataArray[-1][-2]) / 0.000125) # C1 for other -- X3
			keyData[4] = round(float(dataArray[-1][-3]) / 0.000125) # C2 for right -- X2
			keyData[5] = round(float(dataArray[-1][-1]) / 0.000125) # C3 for left -- X4
		#print(keyData)
		return keyData

if __name__ == "__main__":
	obtainKeyData(filePath)

