#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

__author__ = 'Mason Zhang'

# Name: autoXmlFilter.py

import os
import os.path
from xmlFilter import obtainKeyData

targetDir = r'H:\selectXml'

for parent, dirs, filenames in os.walk(targetDir):
		for file in filenames:
			#print(file)
			targetFile = os.path.join(targetDir, file)
			alist = obtainKeyData(targetFile)
			if alist != 'Bad data!':
				print('\t %d '*6 % tuple(alist))