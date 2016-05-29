#!/usr/bin/python
#coding=utf-8
"""
Recovering the output of LSTM to readable form
"""
import sys
import traceback
import os

def main():
	testsentences=[]
	#load test sentence (answer)
	subdirs=os.listdir(WSJDATA_DIR)
	for subdir in subdirs:
		#basename=subdir.lstrip("0")
		dataid=int(subdir)
		if(dataid<fromdataid):
			continue
		if(dataid>todataid and todataid>0):
			continue
		dirpath=WSJDATA_DIR+"/"+subdir
		fs=os.listdir(dirpath)
		for f in fs:
			fp=dirpath+"/"+f
			fi=open(fp,"r")
			for line in fi:
				line=line.strip()
				if(line==""):
					continue
				seninfo=[]
				toks=line.split(" ")
				for tok in toks:
					ts=tok.split("/")
					word=ts[0]
					tag=ts[1]
					seninfo.append(word+"\t"+tag)
				testsentences.append(seninfo)

	endp=len(tagdict)-1
	outputdim=len(tagdict)
	fi=open(predict_output)
	predictsens=[]
	for line in fi:
		line=line.strip()
		toks=line.split(";")
		values=[float(i) for i in toks[1:]]
		senlabels=[]
		maxid=0
		maxvalue=-1
		for i in range(len(values)):
			v=values[i]
			if(v>maxvalue):
				maxvalue=v
				maxid=i
			if(i%outputdim==endp):
				label=maxid%outputdim
				senlabels.append(label)
				maxvalue=-1
		predictsens.append(senlabels)
	fi.close()
	assert len(predictsens)==len(testsentences)
	fo=open(recover_output,"w")
	flag=0
	wordnum=0
	sennum=0
	for i in range(len(testsentences)):
		testitem=testsentences[i]
		predictitem=predictsens[i]
		assert len(testitem)==len(predictitem)
		if(flag==0): #output all
			for j in range(len(testitem)):
				fo.write(testitem[j]+"\t"+tagdict[predictitem[j]]+"\n")
				wordnum+=1
			fo.write("\n")
			sennum+=1
		else: #merge those from the same word
			lastwid="-1"
			lastword=""
			lastwordpos=""
			lastwordpos_p=""
			for j in range(len(testitem)):
				testline=testitem[j]
				toks=testline.split("\t")
				word=toks[0]
				pos=toks[1]
				wid=toks[2]
				p_pos=tagdict[predictitem[j]]
				if(wid!=lastwid and lastword!=""):
					fo.write(lastword+"\t"+lastwordpos+"\t"+lastwid+"\t"+lastwordpos_p+"\n")
					lastword=""
					wordnum+=1
				if(lastword==""):
					lastword=word
				else:
					lastword+="-"+word
				lastwordpos=pos
				lastwordpos_p=p_pos
				lastwid=wid
			if(lastword!=""):
				fo.write(lastword+"\t"+lastwordpos+"\t"+lastwid+"\t"+lastwordpos_p+"\n")
				wordnum+=1
			fo.write("\n")
			sennum+=1
	fo.close()
	print "wordnum: %d"%wordnum
	print "sennum: %d"%sennum

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

def get_configvalue(keyname):
	if(keyname in CONFIG):
		return CONFIG[keyname]
	return ""

def loaddict(dictfile):
	tid=0
	tdict={}
	fi=open(dictfile)
	for line in fi:
		line=line.strip()
		tdict[tid]=line
		tid+=1
	fi.close()
	return tdict

if __name__ == '__main__':
	if(len(sys.argv)!=2):
		print >> sys.stderr, "USAGE: %s configlog"%(__file__)
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),2)
		exit(1)
	try:
		configlog=sys.argv[1]
		CONFIG={}
		loadconfig()
		recover_output=CONFIG["recover_output"]
		predict_output=CONFIG["predict_output"]
		TAG_DICT=CONFIG["TAG_DICT"]
		WSJDATA_DIR=CONFIG["WSJDATA_DIR"]
		TESTDATA_RANGE=CONFIG["TESTDATA_RANGE"]
		fromdataid, todataid=get_sectionrange(TESTDATA_RANGE)
		tagdict=loaddict(TAG_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
