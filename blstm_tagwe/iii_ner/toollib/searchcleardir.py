#!/usr/bin/python
#coding=utf-8
#remove the specific files or dirs
import os
import shutil

rootdir="."
rmdirs=["tmp","model"]

def main():
	 searchcleardir(rootdir)

def searchcleardir(targetdir):
	items=os.listdir(targetdir)
	for item in items:
		path=targetdir+"/"+item
		if(os.path.isdir(path)):
			if(item in rmdirs):
				cleardir(path)
				print "clear: %s"%path
				#os.rmdir(path)
			else:
				searchcleardir(path)

def cleardir(targetdir):
	for root,dirs,files in os.walk(targetdir, topdown=False):
		for name in files:
			os.remove(os.path.join(root,name))
		for name in dirs:
			os.rmdir(os.path.join(root,name))

def cleardir2(targetdir):
	items=os.listdir(targetdir)
	for item in items:
		path=targetdir+"/"+item
		if(os.path.isfile(path)):
			os.remove(path)
		elif(os.path.isdir(path)):
			cleardir2(path)
			os.rmdir(path)
		else:
			raise Exception, "unknown file type: %s"%path
		
def deletedir(targetdir):
	if(not os.path.exists(targetdir)):
		return
	shutil.rmtree(targetdir)

if __name__ == '__main__':
	main()

