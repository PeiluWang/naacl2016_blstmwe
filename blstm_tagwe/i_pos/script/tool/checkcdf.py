#!/usr/bin/python
#coding=utf-8

from netCDF4 import Dataset as ds
import numpy as np
import sys

if(len(sys.argv)<=1):
	f1="train.nc"
else:
	f1=sys.argv[1]

print "checking netCDF file: "+f1

nc1=ds(f1,'r',format='NETCDF4')

print "\n-------dimension-------\n"
for dim in nc1.dimensions:
	print nc1.dimensions[dim]

print "\n-------variable metainfo-------\n"
for var in nc1.variables:
	print var+":",
	print nc1.variables[var].ndim, #data dim
	print nc1.variables[var].shape #shape

print "\n-------seq names-------\n"
print nc1.variables["seqTags"][:]

print "\n-------seq lengths-------\n"
print nc1.variables["seqLengths"][:]

print "\n-------inputFeats-------\n"
#print nc1.variables["inputs"][0:3,0:2] #print part of the matrix
print nc1.variables["inputFeats"][:,:] #print all

print "\n-------inputWords-------\n"
#print nc1.variables["inputs"][0:3,0:2] #print part of the matrix
print nc1.variables["inputWords"][:] #print all


print "\n-------outputs-------\n"
#print nc1.variables["targetPatterns"][0:3,0:2] #print part of the matrix
print nc1.variables["outputLabels"][0:200] #print all






nc1.close()
