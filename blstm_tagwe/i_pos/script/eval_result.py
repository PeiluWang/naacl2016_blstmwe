#!/usr/bin/python
#coding=utf-8
"""
Recovering the output of LSTM to readable form
"""
import sys
import traceback


def cal_accu():
	total_num=0
	correct_num=0
	sen_num=0
	sen_correct=0
	sen_allcorrect=True
	fi=open(recover_output)
	for line in fi:
		line=line.strip()
		if(line==""):
			if(sen_allcorrect):
				sen_correct+=1
			sen_num+=1
			sen_allcorrect=True
			continue
		total_num+=1
		toks=line.split("\t")
		if(toks[1]==toks[-1]):
			correct_num+=1
		else:
			sen_allcorrect=False
	fi.close()
	error_num=total_num-correct_num
	print "total_num: %d"%total_num
	print "correct_num: %d"%correct_num
	print "accuracy: %f"%(float(correct_num)/total_num)
	print "error rate: %f"%(float(error_num)/total_num)
	print "sen_num: %d"%sen_num
	print "sen_correct: %d"%sen_correct
	print "sen correct rate: %f"%(float(sen_correct)/sen_num)
	fo=open(evalresult_file,"w")
	fo.write("total_num: %d\n"%total_num)
	fo.write("correct_num: %d\n"%correct_num)
	fo.write("accuracy: %f\n"%(float(correct_num)/total_num))
	fo.write("error rate: %f\n"%(float(error_num)/total_num))
	fo.write( "sen_num: %d\n"%sen_num)
	fo.write( "sen_correct: %d\n"%sen_correct)
	fo.write( "sen correct rate: %f\n"%(float(sen_correct)/sen_num))

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
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),2)
		exit(1)
	try:
		configlog=sys.argv[1]
		CONFIG={}
		loadconfig()
		recover_output=CONFIG["recover_output"]
		evalresult_file=CONFIG["evalresult_file"]
		cal_accu()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
