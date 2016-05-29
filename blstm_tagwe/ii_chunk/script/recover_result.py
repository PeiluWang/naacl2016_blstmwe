#!/usr/bin/python
#coding=utf-8
"""
Recovering the output of LSTM to readable form
"""
import sys
import traceback


def main():
	testsentences=[]
	fi=open(TESTDATA)
	sen=[]
	for line in fi:
		line=line.strip()
		if(line==""):
			if(len(sen)>0):
				testsentences.append(sen)
				sen=[]
			continue
		sen.append(line)
	fi.close()
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
	for i in range(len(testsentences)):
		testitem=testsentences[i]
		predictitem=predictsens[i]
		assert len(testitem)==len(predictitem)
		for j in range(len(testitem)):
			fo.write(testitem[j]+" "+tagdict[predictitem[j]]+"\n")
		fo.write("\n")
	fo.close()



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
		TESTDATA=CONFIG["TESTDATA"]
	
		tagdict=loaddict(TAG_DICT)
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
