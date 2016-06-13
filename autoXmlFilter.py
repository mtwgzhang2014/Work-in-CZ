#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

__author__ = 'Mason Zhang'

# Name: autoXmlFilter.py

import os
import os.path
from xmlFilter import obtainKeyData

#targetDir = r'H:\selectXml'
targetDir = r'C:\Users\zhang-384\Desktop\My Works\00 OST\2015\2. 钢自动角差项目\selectXml'

for parent, dirs, filenames in os.walk(targetDir):
		for file in filenames:
			#print(file)
			targetFile = os.path.join(targetDir, file)
			alist = obtainKeyData(targetFile)
			if alist != 'Bad data!':
				print('\t %d '*6 % tuple(alist))