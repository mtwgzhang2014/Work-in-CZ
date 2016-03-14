#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

__author__ = 'Mason Zhang'

# Name: xmlFilter.py 
# Purpose: 解析XML文件，提取并处理角差记录数据
#  1. 提取XML中的特定数据，DATA中数据部分内容
#  2. 将DATA中第5行到第12行数据放入一个二维数组中
#  3. 记录第一行的初始角差值 
#  4. 最终计算出每个XML文件中对各角的最终进刀量

from xml.parsers.expat import ParserCreate

filePath = r'C:\Users\zhang-384\Desktop\7284080391-6-20150320121642-MeasData.xml'

class DefaultSaxHandler(object):

	elementName = ''
	dataContent = ''

	def start_element(self, name, attrs):
		elementName = name
        #print('sax:start_element: %s, attrs: %s' % (name, str(attrs)))

	def end_element(self, name):
		pass 
        # print('sax:end_element: %s' % name)

	def char_data(self, text):
		if self.elementName == 'DATA':
			self.dataContent = text
			print(text)
        # print('sax:char_data: %s' % text)

# 当SAX解析器读到一个节点时，会产生3个事件：
# start_element, char_data, end_element

handler = DefaultSaxHandler()
parser = ParserCreate()

# 注册事件处理程序
parser.StartElementHandler = handler.start_element
parser.EndElementHandler = handler.end_element
parser.CharacterDataHandler = handler.char_data
# parser.Parse(xml)

with open(filePath, 'r') as xmlfile:
	xmltmp = xmlfile.read()
	parser.Parse(xmltmp)
	print(handler.dataContent)


