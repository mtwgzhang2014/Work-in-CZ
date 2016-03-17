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

with open(filePath, 'r') as xmlfile:
	xmltmp = xmlfile.read()
	# re.search 扫描整个字符串并返回第一个成功的匹配
	# 而re.match 尝试从字符串的起始位置匹配一个模式，
	# 如果不是起始位置匹配成功的话，match()就返回none
	data = re.search(pattern, xmltmp, flags = 0)
	if data != None:
		print(data.group(1))
	else:
		print(data)


