#!/usr/bin/python
#coding=utf-8
"""
Evaluate the output of the currennt
Calculate the ppl
"""
import sys
import os
import struct
import traceback
import numpy as np
import math

CONFIG={}

def	main():
	#calculate ppl
	seninfos=[]
	fi=open(TEST_DATA,"r")
	lineid=0
	for line in fi:
		line=line.strip()
		words=line.split(" ")
		wordnum=len(words)
		linename="%010d"%lineid
		seninfos.append([linename,wordnum,words])
		lineid+=1
	fi.close()
	fo=open(evalresult_file,"w")
	fi=open(predict_output,"r")
	lineid=0
	sum_ppl=0
	for line in fi:
		line=line.strip()
		toks=line.split(";")
		assert toks[0]==seninfos[lineid][0]
		assert len(toks)-1==seninfos[lineid][1]*output_dim
		words=seninfos[lineid][2]
		wid=0
		logp=0
		for word in words:
			wordid=wordid_dict[word]
			p=float(toks[1+output_dim*wid+wordid])
			logp+=math.log(p,2)
		avg_logp=logp/len(words)
		ppl=2**(-avg_logp)
		fo.write(toks[0]+"\t%d"%seninfos[lineid][1]+"\t%f"%ppl+"\n")
		sum_ppl+=ppl
		lineid+=1
		if(lineid%100==0):
			print lineid
	avg_ppl=sum_ppl/(lineid)
	fo.write("total line: %d"%lineid+"\n")
	fo.write("avg ppl: %d"%avg_ppl+"\n")

def load_worddict(dictfile):
	wordid=0
	wordid_dict={}
	fi=open(dictfile,"r")
	for line in fi:
		line=line.strip()
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
	if(len(sys.argv)!=2):
		print >> sys.stderr, "USAGE: %s configlog"%(__file__)
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),3)
		exit(1)
	try:
		configlog=sys.argv[1]
		loadconfig()
		#load parameters
		WORD_DICT=CONFIG["WORD_DICT"]
		output_dim=int(CONFIG["output_dim"])
		TEST_DATA=CONFIG["TEST_DATA"]
		predict_output=CONFIG["predict_output"]
		evalresult_file=CONFIG["evalresult_file"]
		#get fromdata id and todata id
		wordid_dict=load_worddict(WORD_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
