#!/usr/bin/python
#coding=utf-8
"""
Recovering the output of LSTM to readable form
然后把iobes转成iob格式
之后使用viterbi解码
"""
import sys
import traceback
import math

transmatrix=[]
transtagiddict={}

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

	assert outputdim==len(transtagiddict)
	assert outputdim==len(transmatrix)
	assert outputdim==len(transmatrix[0])

	fi=open(predict_output)
	predictsens=[]
	labelset=set()
	for line in fi:
		#print line
		line=line.strip()
		toks=line.split(";")
		values=[float(i) for i in toks[1:]]
		senlabels=[]
		#get predict label
		#using viterbi algorithm
		maxid=0
		maxvalue=-1
		bestscores=[]
		trackbacks=[]
		nnoutputdists=[]
		dist=[]
		sump=0
		
		for i in range(len(values)):
			v=values[i]
			dist.append(v)
			sump+=v
			if(i%outputdim==endp):
				assert abs(sump-1)<0.001
				sump=0
				nnoutputdists.append(dist)
				dist=[]
		steps=len(nnoutputdists)

		for i in range(steps):
			dist=nnoutputdists[i]
			if(i==0):
				bestscore=[]
				for j in range(outputdim):
					bestscore.append(math.log(dist[j]))
				bestscores.append(bestscore)
				traceback=[]
				for j in range(outputdim):
					traceback=-1
				trackbacks.append(traceback)
			else:
				pbestscorelist=bestscores[-1]
				bestscorelist=[]
				trackbacklist=[]
				for cid in range(outputdim): #current id
					bestscore=-1
					besttrack=-1
					firstflag=True
					#print cid
					for pid in range(outputdim): #prev id
						pbestscore=pbestscorelist[pid]
						A=transmatrix[pid][cid]
						#score=pbestscore
						if(A>0.000001):
							score=pbestscore#+math.log(A)
						else:
							continue
						#print score
						#print pbestscore
						if(firstflag):
							bestscore=score
							besttrack=pid
							firstflag=False
							continue
						if(bestscore<score):
							bestscore=score
							besttrack=pid
					#print ""
					assert besttrack>=0
					bestscore=bestscore+math.log(dist[cid])
					bestscorelist.append(bestscore)
					trackbacklist.append(besttrack)
				bestscores.append(bestscorelist)
				trackbacks.append(trackbacklist)

		trackbacklabels=[]
		for i in range(steps):
			j=steps-i-1
			if(j==steps-1):
				bestscore=-1
				bestlabel=-1
				firstflag=True
				bestscorelist=bestscores[j]
				for cid in range(outputdim):
					score=bestscorelist[cid]
					if(firstflag):
						bestscore=score
						bestlabel=cid
						firstflag=False
						continue
					if(bestscore<score):
						bestscore=score
						bestlabel=cid
				assert bestlabel>=0
				trackbacklabels.append(bestlabel)
			else:
				lasttrackbacklist=trackbacks[j+1]
				lasttracklabel=trackbacklabels[-1]
				cid=lasttrackbacklist[lasttracklabel]
				trackbacklabels.append(cid)

		assert len(trackbacklabels)==steps
		trackbacklabels.reverse()
		senlabels=trackbacklabels
		#print senlabels

		predictsens.append(senlabels)
		#break
	fi.close()
	#exit(0)

	assert len(predictsens)==len(testsentences)
	fo=open(recover_output,"w")
	for i in range(len(testsentences)):
		testitem=testsentences[i]
		predictitem=predictsens[i]
		assert len(testitem)==len(predictitem)
		
		lasttagtype=""
		for j in range(len(testitem)):
			predicttag=tagdict[predictitem[j]]
			testline=testitem[j]
			toks=testline.split(" ")
			answertag=toks[-2]
			seninfoline=testline+" "+predicttag+" "+answertag
			if(predicttag=="O"):
				fo.write(seninfoline+" O\n")
				lasttagtype="O"
				continue
			ts=predicttag.split("-")
			tagstat=ts[0]
			tagtype=ts[1]
			if(tagstat=="B" or tagstat=="S"):
				if(tagtype!=lasttagtype):
					fo.write(seninfoline+" I-"+tagtype+"\n")
				else:
					fo.write(seninfoline+" B-"+tagtype+"\n")
			else:
				fo.write(seninfoline+" I-"+tagtype+"\n")
			lasttagtype=tagtype
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

def loadtransmatrix():
	global transmatrix
	global transtagiddict
	fi=open(transmatrix_file)
	id=0
	for line in fi:
		line=line.strip()
		toks=line.split("\t")
		tag=toks[0]
		transtagiddict[tag]=id
		id+=1
		row=[]
		for item in toks[1:]:
			row.append(float(item))
		transmatrix.append(row)
	fi.close()

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
		transmatrix_file=CONFIG["transmatrix_file"]
	
		tagdict=loaddict(TAG_DICT)
		loadtransmatrix()
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
