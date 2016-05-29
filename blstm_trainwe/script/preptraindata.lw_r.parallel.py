#!/usr/bin/python
#coding=utf-8
"""
Prepare the autoencoder data
按照collobert版本处理，随机替换一定比率的词，其tag为0

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
import random
import multiprocessing

CONFIG={}

random.seed(0)

def main():
	if(preptype=="train"):
		lock=multiprocessing.Lock()
		subprocesses=[]
		for fid in range(prep_fraction_startid, prep_fraction_startid+thread_num):
			p=multiprocessing.Process(target=subprocess, args=(lock, fid,thread_num,prep_fraction_endid,\
				WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate))
			p.start()
			subprocesses.append(p)
		for p in subprocesses:
			p.join()
	else:
		dealonefraction(-1,WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate)

def main2():
	if(preptype=="train"):
		for i in range(prep_fraction_startid,prep_fraction_endid+1):
			print "dealing fraction: %d"%i
			dealonefraction(i,WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate)
	else:
		dealonefraction(-1,WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate)

def subprocess(lock, fractionid, parallelnum, maxid, WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate):
	for fid in range(fractionid, maxid+1,  parallelnum):
		lock.acquire()
		print "dealing %d"%fid
		lock.release()
		dealonefraction(fid,WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate)


def	dealonefraction(fractionid,WORD_DICT,prep_fraction_size,DATA_FILE,output_file,replacerate):
	#print "deal fraction begin: %d"%fractionid
	#random.seed(0)
	#load data
	wordid_dict=load_lowerworddict(WORD_DICT)
	maxwordid=len(wordid_dict)-1
	#get basic information
	seq_num=0
	step_num=0
	seqname_maxlength=10
	input_dim=len(wordid_dict) # 1 additional dim is for startflag
	feat_dim=1
	output_dim=2 #additional tag ot represent the end tag
	
	if(fractionid>=0):
		start_senid=fractionid*prep_fraction_size
		end_senid=(fractionid+1)*prep_fraction_size
	else:
		start_senid=-1
		end_senid=-1

	fi=open(DATA_FILE,"r")
	ln=0
	for line in fi:
		if(ln<start_senid and start_senid>=0):
			ln+=1
			continue
		if(ln>=end_senid and end_senid>=0):
			break
		ln+=1
		line=line.strip()
		if(line==""):
			continue
		senwords=line.split(" ")
		step_num+=len(senwords)
		seq_num+=1
	fi.close()
	if(seq_num==0):
		print "fraction null: %d"%fractionid
		return
	#define netcdf file
	if(fractionid>=0):
		poutput_file=output_file+"."+str(fractionid)
	else:
		poutput_file=output_file
	nc=ds(poutput_file,"w",format="NETCDF4")
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

	fi=open(DATA_FILE,"r")
	ln=0
	for line in fi:
		if(ln<start_senid and start_senid>=0):
			ln+=1
			continue
		if(ln>=end_senid and end_senid>=0):
			break
		ln+=1
		line=line.strip()
		if(line==""):
			continue
		senwords=line.split(" ")
		if(start_senid>=0):
			seqname="%010d"%(sen_index+start_senid)
		else:
			seqname="%010d"%sen_index
		sample_num=len(senwords)

		for i in range(sample_num):
			r=random.random()
			wordid=-1
			tagid=-1
			if(r<replacerate): #replace with a random word, tag is 0 ,  
				wordid=random.randint(0,maxwordid)
				tagid=0
			else: # original word, tag is 1
				wordid=getwordid(senwords[i], wordid_dict)
				tagid=1

			ncvar_inputwords[frame_index]=wordid
			ncvar_inputfeats[frame_index,0]=0
			ncvar_outputlabels[frame_index]=tagid
			frame_index+=1			

		ncvar_seqnames[sen_index,0:seqname_maxlength]=seqname
		ncvar_seqlengths[sen_index]=sample_num

		sen_index+=1

	nc.close()
	fi.close()

def getwordid(word, wordid_dict):
	word=word.lower()
	if(word in wordid_dict):
		return wordid_dict[word]
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

def load_lowerworddict(dictfile):
	wordid=0
	wordid_dict={}
	fi=open(dictfile,"r")
	for line in fi:
		line=line.strip()
		lword=line.lower()
		assert lword==line
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
		replacerate=float(CONFIG["replacerate"])
		prep_fraction_size=int(CONFIG["prep_fraction_size"])
		prep_fraction_startid=int(CONFIG["prep_fraction_startid"])
		prep_fraction_endid=int(CONFIG["prep_fraction_endid"])
		thread_num=int(CONFIG["thread_num"])
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
		
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
