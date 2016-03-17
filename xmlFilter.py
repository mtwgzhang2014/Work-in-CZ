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

filePath = r'C:\Users\zhang-384\Desktop\7284080391-6-20150320121642-MeasData.xml'
# 获取纯角差数据
pattern = re.compile("Hone_Depth2(.*?)EndTime:", re.S) 
# re.S 使 . 匹配包括换行在内的所有字符
# 在此可以避免换行符\n不能被匹配

with open(filePath, 'r') as xmlfile:
	tmpXml = xmlfile.read()
	# re.search 扫描整个字符串并返回第一个成功的匹配
	# 而re.match 尝试从字符串的起始位置匹配一个模式，
	# 如果不是起始位置匹配成功的话，match()就返回none
	tmpData = re.search(pattern, tmpXml, flags = 0)
	# if tmpData != None:
	# 	print(tmpData.group(1))
	# else:
	# 	print(tmpData)

if tmpData!=None:
	dataList = tmpData.group(1).split()
	g = iter(dataList)
	if len(dataList) % 12 == 0:
		rowNum = int(len(dataList)/12)
		# 下面是两种将一维List转变为二维List的语句
		# 1. 采用迭代器结果逐个赋值到二维List中，[[0 for..] for..]二维数组
		# 2. 利用zip函数的特性，但结果为List中包含tuple的形式
		dataArray = [[next(g) for _ in range(12)] for _ in range(rowNum)]
		# * unpack the arguments out of a list or tuple
		#dataArray = list(zip(*[iter(dataList)]*12))
	elif len(dataList) % 16 == 0:
		pass
	else:
		pass
	# 如何处理即可以被12整除又可以被16整除的情况

	# dataArray[i][j] for i is row and j is column
	print(dataArray)
	print(dataArray[14][11])