#!/usr/bin/python
#coding=utf-8
#生成lstm训练要用的文件
#使用v12类型数据，inputfeats, inputwords
#处理tfs，三列格式数据
import os
import codecs
import sys
import traceback
from netCDF4 import Dataset as ds

def main():
	sentences=[]
	seq_num=0
	step_num=0
	seqname_maxlength=10
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
			seninfo=[]
			for line in fi:
				line=line.strip()
				if(line==""):
					if(len(seninfo)>0):
						sentences.append(seninfo)
					seninfo=[]
					continue
				toks=line.split("\t")
				word=toks[0]
				pos=toks[1]
				wordl=word.lower()
				if(word==wordl):
					lower_flag=0
				else:
					lower_flag=1 # word contains capital letters
				wordid=worddict[wordl]
				tagid=tagdict[pos]
				seninfo.append((wordid,tagid,lower_flag))
				step_num+=1
			fi.close()
	seq_num=len(sentences)

	print "seq_num: %d"%seq_num
	print "step_num: %d"%step_num

	#define netcdf file
	nc=ds(OUTPUT_FILE,"w",format="NETCDF4")
	nc.createDimension("seq_num",seq_num)
	nc.createDimension("step_num",step_num)
	nc.createDimension("feat_dim",feat_dim)
	nc.createDimension("output_dim",output_dim)
	nc.createDimension("seqname_maxlength",seqname_maxlength)
	
	ncvar_seqnames=nc.createVariable("seqTags","c",("seq_num","seqname_maxlength"))
	ncvar_seqlengths=nc.createVariable("seqLengths","i4",("seq_num"))
	ncvar_inputfeats=nc.createVariable("inputFeats","f4",("step_num","feat_dim"))
	ncvar_inputwords=nc.createVariable("inputWords","i4",("step_num"))
	ncvar_outputlabels=nc.createVariable("outputLabels","i4",("step_num"))

	frame_index=0
	sen_index=0
	for senwords in sentences:
		seqname="%010d"%sen_index
		sample_num=len(senwords) 
		for i in range(sample_num):
			wordid=senwords[i][0]
			tagid=senwords[i][1]
			lower_flag=senwords[i][2]
			ncvar_inputwords[frame_index]=wordid
			ncvar_inputfeats[frame_index,0]=lower_flag
			ncvar_outputlabels[frame_index]=tagid
			frame_index+=1

		ncvar_seqnames[sen_index,0:seqname_maxlength]=seqname
		ncvar_seqlengths[sen_index]=sample_num

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
		feat_dim=int(CONFIG["inputfeat_dim"])
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