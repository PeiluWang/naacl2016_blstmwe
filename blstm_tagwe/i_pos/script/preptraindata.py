#!/usr/bin/python
#coding=utf-8
#生成lstm训练要用的文件
import os
import codecs
import sys
import traceback
from netCDF4 import Dataset as ds

def main():
	sentences=[]
	seq_num=0
	step_num=0
	filename_maxlength=10
	input_dim=1
	output_dim=len(tagdict)
	subdirs=os.listdir(WSJDATA_DIR)
	for subdir in subdirs:
		#basename=subdir.lstrip("0")
		dataid=int(subdir)
		if(dataid<fromdataid):
			continue
		if(dataid>todataid and todataid>0):
			continue
		dirpath=WSJDATA_DIR+"/"+subdir
		#print subdir
		fs=os.listdir(dirpath)
		for f in fs:
			fp=dirpath+"/"+f
			fi=open(fp,"r")
			for line in fi:
				line=line.strip()
				if(line==""):
					continue
				toks=line.split(" ")
				seninfo=[]
				for tok in toks:
					ts=tok.split("/")
					wordid=worddict[ts[0]]
					tagid=tagdict[ts[1]]
					seninfo.append((wordid,tagid))
				sentences.append(seninfo)
				step_num+=len(toks)
			fi.close()
	seq_num=len(sentences)
	#define netcdf file
	nc=ds(OUTPUT_FILE,"w",format="NETCDF4")
	nc.createDimension("numSeqs",seq_num)
	nc.createDimension("numTimesteps",step_num)
	nc.createDimension("inputPattSize",input_dim)
	nc.createDimension("numLabels",output_dim)
	nc.createDimension("maxSeqTagLength",filename_maxlength)
	
	ncvar_filenames=nc.createVariable("seqTags","c",("numSeqs","maxSeqTagLength"))
	ncvar_samplenums=nc.createVariable("seqLengths","i4",("numSeqs"))
	ncvar_inputs=nc.createVariable("inputs","i4",("numTimesteps"))
	ncvar_outputs=nc.createVariable("targetClasses","i4",("numTimesteps"))

	frame_index=0
	sen_index=0
	for senwords in sentences:
		seqname="%010d"%sen_index
		sample_num=len(senwords) 
		for i in range(sample_num):
			wordid=senwords[i][0]
			tagid=senwords[i][1]
			ncvar_inputs[frame_index]=wordid
			ncvar_outputs[frame_index]=tagid
			frame_index+=1

		ncvar_filenames[sen_index,0:filename_maxlength]=seqname
		ncvar_samplenums[sen_index]=sample_num

		sen_index+=1
	nc.close()

def get_sectionrange(datarange):
	toks=datarange.split("~")
	assert len(toks)==2
	fromdataid=int(toks[0])
	if(toks[1]==""):
		todataid=0
	else:
		todataid=int(toks[1])
	return fromdataid,todataid


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

def loaddict(dictfile):
	tid=0
	tdict={}
	fi=open(dictfile)
	for line in fi:
		line=line.strip()
		tdict[line]=tid
		tid+=1
	fi.close()
	return tdict

if __name__ == '__main__':
	if(len(sys.argv)!=3):
		print >> sys.stderr, "USAGE: %s configlog"%(__file__)
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),3)
		exit(1)
	try:
		configlog=sys.argv[1]
		preptype=sys.argv[2] #train, val, test
		CONFIG={}
		loadconfig()
		WSJDATA_DIR=CONFIG["WSJDATA_DIR"]
		WORD_DICT=CONFIG["WORD_DICT"]
		TAG_DICT=CONFIG["TAG_DICT"]
		#load parameters
		if(preptype=="train"):
			DATA_RANGE=CONFIG["TRAINDATA_RANGE"]
			OUTPUT_FILE=CONFIG["train_data"]
		elif(preptype=="val"):
			DATA_RANGE=CONFIG["VALDATA_RANGE"]
			OUTPUT_FILE=CONFIG["val_data"]
		elif(preptype=="test"):
			DATA_RANGE=CONFIG["TESTDATA_RANGE"]
			OUTPUT_FILE=CONFIG["test_data"]
		else:
			raise Exception, "type is not train,val or test: "+preptype

		fromdataid, todataid=get_sectionrange(DATA_RANGE)
		worddict=loaddict(WORD_DICT)
		tagdict=loaddict(TAG_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)