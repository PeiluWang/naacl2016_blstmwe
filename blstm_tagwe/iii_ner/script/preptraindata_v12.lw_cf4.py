#!/usr/bin/python
#coding=utf-8
#生成lstm ner训练要用的文件
#使用v12b类型数据
#输入为lowercase word
#使用4维capital feature，完全仿照senna： i
#allcaps or initcaps or hascaps or nocaps
import os
import codecs
import sys
import traceback
import numpy
from netCDF4 import Dataset as ds

wordcount=0
unkwordcount=0

def main():
	sentences=[]
	seq_num=0
	step_num=0

	assert feat_dim==4

	seqname_maxlength=10
	output_dim=len(tagdict)
	fi=open(INPUT_DATA)
	sen=[]
	for line in fi:
		line=line.strip()
		if(line==""):
			if(len(sen)>0):
				sentences.append(sen)
			sen=[]
			continue
		toks=line.split(" ")
		sen.append(toks)
		step_num+=1

	seq_num=len(sentences)
	#print "step_num: %d"%step_num
	#print "seq_num: %d"%seq_num
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
	for seninfo in sentences:
		seqname="%010d"%sen_index
		sample_num=len(seninfo) 
		for i in range(sample_num):
			word=seninfo[i][0]
			pos=seninfo[i][1]
			chunk=seninfo[i][2]
			tag=seninfo[i][4]

			wordl=word.lower()
			wordid=getwordid(wordl)
			tagid=tagdict[tag]

			allcaps=word.isupper()
			initcap=word[0].isupper()
			hascap=False
			if(not allcaps):
				for w in word:
					if(w.isupper()):
						hascap=True
						break
			else:
				hascap=True

			ncvar_inputwords[frame_index]=wordid

			ncvar_inputfeats[frame_index,:]=numpy.zeros(4)
			if(allcaps):
				ncvar_inputfeats[frame_index,0]=1
			elif(initcap):
				ncvar_inputfeats[frame_index,1]=1
			elif(hascap):
				ncvar_inputfeats[frame_index,2]=1
			else:
				ncvar_inputfeats[frame_index,3]=1

			ncvar_outputlabels[frame_index]=tagid
			frame_index+=1

		ncvar_seqnames[sen_index,0:seqname_maxlength]=seqname
		ncvar_seqlengths[sen_index]=sample_num
		sen_index+=1
	nc.close()

	print "wordcount: %d"%wordcount
	print "unkwordcount: %d"%unkwordcount
	print "oov rate: %f"%(float(unkwordcount)/wordcount)


def getwordid(word):
	global wordcount
	global unkwordcount
	wordcount+=1
	if(word in worddict):
		return worddict[word]
	unkwordcount+=1
	print word
	exit(1)
	return worddict["<unk>"]

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
		WORD_DICT=CONFIG["WORD_DICT"]
		TAG_DICT=CONFIG["TAG_DICT"]
		feat_dim=int(CONFIG["inputfeat_dim"])
		#load parameters
		if(preptype=="train"):
			INPUT_DATA=CONFIG["TRAINDATA"]
			OUTPUT_FILE=CONFIG["train_data"]
		elif(preptype=="val"):
			INPUT_DATA=CONFIG["VALDATA"]
			OUTPUT_FILE=CONFIG["val_data"]
		elif(preptype=="test"):
			INPUT_DATA=CONFIG["TESTDATA"]
			OUTPUT_FILE=CONFIG["test_data"]
		else:
			raise Exception, "type is not train,val or test: "+preptype

		worddict=loaddict(WORD_DICT)
		tagdict=loaddict(TAG_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)