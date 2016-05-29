#!/usr/bin/python
#coding=utf-8
"""
Prepare the autoencoder data
Convert the corpus to the input and output data of LM
没有出现的词，当做<unk>处理
所有词小写
处理大数据版本，不在内存中保留sentence数据，而是重新读取
"""
import sys
import os
import struct
import traceback
import numpy as np
from netCDF4 import Dataset as ds

CONFIG={}

total_wordnum=0
unk_wordnum=0

def	main():
	#get basic information
	seqs_num=0
	step_num=0
	filename_maxlength=10
	input_dim=len(wordid_dict) # 1 additional dim is for startflag
	output_dim=len(wordid_dict) #additional tag ot represent the end tag
	starttagid=len(wordid_dict) #the start tag id
	fi=open(DATA_FILE,"r")
	for line in fi:
		line=line.strip()
		if(line==""):
			continue
		senwords=line.split(" ")
		step_num+=len(senwords)
		seqs_num+=1
	fi.close()
	#define netcdf file
	nc=ds(output_file,"w",format="NETCDF4")
	nc.createDimension("numSeqs",seqs_num)
	nc.createDimension("numTimesteps",step_num)
	nc.createDimension("inputPattSize",input_dim)
	nc.createDimension("maxSeqTagLength",filename_maxlength)
	nc.createDimension("numLabels",output_dim)

	ncvar_filenames=nc.createVariable("seqTags","c",("numSeqs","maxSeqTagLength"))
	ncvar_samplenums=nc.createVariable("seqLengths","i4",("numSeqs"))
	ncvar_inputs=nc.createVariable("inputs","i4",("numTimesteps"))
	ncvar_outputs=nc.createVariable("targetClasses","i4",("numTimesteps"))

	frame_index=0
	sen_index=0
	
	fi=open(DATA_FILE,"r")
	for line in fi:
		line=line.strip()
		if(line==""):
			continue
		senwords=line.split(" ")
		seqname="%010d"%sen_index
		sample_num=len(senwords) 

		for i in range(sample_num):
			wordid=getwordid(senwords[i])
			ncvar_inputs[frame_index]=wordid
			ncvar_outputs[frame_index]=wordid
			frame_index+=1

		ncvar_filenames[sen_index,0:filename_maxlength]=seqname
		ncvar_samplenums[sen_index]=sample_num

		sen_index+=1
		#if(sen_index%1000==0):
		#	print sen_index
			#break

	nc.close()
	fi.close()

	print "worddict size: %d"%len(wordid_dict)
	print "total_wordnum: %d"%total_wordnum
	print "unk_wordnum: %d"%unk_wordnum
	print "oov rate: %f"%(float(unk_wordnum)/total_wordnum)


def getwordid(word):
	global total_wordnum
	global unk_wordnum
	word=word.lower()
	total_wordnum+=1
	if(word in wordid_dict):
		return wordid_dict[word]
	unk_wordnum+=1
	#raise Exception, "unk word! "+word
	return wordid_dict["<unk>"]

def initdir(dirpath):
	if(os.path.exists(dirpath)):
		#clean the dir
		fs=os.listdir(dirpath)
		for f in fs:
			os.remove(dirpath+"/"+f)
		return
	os.mkdir(dirpath)

def load_worddict(dictfile):
	wordid=0
	wordid_dict={}
	fi=open(dictfile,"r")
	for line in fi:
		line=line.strip()
		line=line.lower()
		wordid_dict[line]=wordid
		wordid+=1
	fi.close()
	return wordid_dict
	
#load the exp config
def loadconfig():
	global CONFIG
	fi=open(configlog,"r")
	for line in fi:
		line=line.strip()
		if(line.startswith("#")):
			continue
		if(line==""):
			continue
		toks=line.split("=")
		if(len(toks)!=2):
			continue
		CONFIG[toks[0].strip()]=toks[1].strip()
	fi.close()

if __name__ == '__main__':
	if(len(sys.argv)!=3):
		print >> sys.stderr, "USAGE: %s configlog"%(__file__)
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),3)
		exit(1)
	try:
		configlog=sys.argv[1]
		preptype=sys.argv[2] #train, val, test
		loadconfig()
		#load parameters
		WORD_DICT=CONFIG["WORD_DICT"]
		if(preptype=="train"):
			DATA_FILE=CONFIG["TRAIN_DATA"]
			output_file=CONFIG["train_data"]
		elif(preptype=="val"): 
			DATA_FILE=CONFIG["VAL_DATA"]
			output_file=CONFIG["val_data"]
		elif(preptype=="test"):
			DATA_FILE=CONFIG["TEST_DATA"]
			output_file=CONFIG["test_data"]
		else:
			raise Exception, "type is not train,val or test: "+preptype
		#get fromdata id and todata id
		wordid_dict=load_worddict(WORD_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
