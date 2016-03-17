#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Mason Zhang'

# Name: selectFile.py 
# Purpose: 从指定文件夹中挑选文件包含特殊字符串的文件并复制到指定文件夹

import os
import os.path
import shutil

# sourceDir = r'H:\xml backup'
# targetDir = r'H:\selectXml'

sourceDir = r'C:\Users\zhang-384\Desktop\My Works\ProjectData\AutoHoning4SteelLC\xml backup'
targetDir = r'C:\Users\zhang-384\Desktop\My Works\ProjectData\AutoHoning4SteelLC\selectXml'


def selectFile(sourceDir, targetDir, searchKey):
	if not os.path.exists(targetDir):
		os.makedirs(targetDir)

	totalFile = sum([len(files) for root, dirs, files in os.walk(sourceDir)])	

	checkedFile = 0.0

	for parent, dir, filenames in os.walk(sourceDir):
		for file in filenames:
			if searchKey in file:
				sourceFile = os.path.join(sourceDir, file)
				targetFile = os.path.join(targetDir, file)
				shutil.copy2(sourceFile, targetFile)
			checkedFile = checkedFile + 1
			# 动态显示搜索进度
			print('|'*(round(checkedFile/totalFile*80)) + ' %2f' % (round(checkedFile/totalFile*100,2)))

	print('Finish!')

if __name__ == "__main__":
	selectFile(sourceDir, targetDir, 'MeasData')