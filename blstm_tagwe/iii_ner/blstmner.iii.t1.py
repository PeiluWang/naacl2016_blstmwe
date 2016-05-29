#!/usr/bin/python
#coding=utf-8
import os
import sys
import traceback
import types
import socket
if("toollib" not in sys.path):
	sys.path.append("toollib")
import checkgpu
"""
Use BLTM to do NER tagging
"""
##############################
#Configuration
EXP_NAME=__file__.rstrip(".py")
ROOT_DIR="exp/%s"%EXP_NAME
TRAINDATA="data/eng.train_rf.iobes"
VALDATA="data/eng.testa_rf.iobes"
TESTDATA="data/eng.testb_rf.iobes"
transmatrix_file="data/eng.train_rf.transmatrix.t"
#config file
NETWORK_CFG="resource/network.i50b200o17.jsn"
TRAIN_CFG_TEMPLATE="resource/exp.train.template.cfg"
TEST_CFG_TEMPLATE="resource/exp.test.template.cfg"
WORD_DICT="data/lowerworddict.nerrf.txt"
TAG_DICT="data/tagdict.iobes.txt"
load_weweights = "none"
load_featweights = "none"
load_lstmweights = "none"
#tool
SCRIPT_DIR="script"
RNN_TOOL="../../tool_rnn/currennt_v12c/currennt/Release/currennt.exe"
#log
CONFIG_LOG=ROOT_DIR+"/config.log"
#############################
#Step Switches
#prepare data
z_PREP_DATA		= 0
#train and test
z_PREP_CFG              	= 1  #  Gnerate config file (this switch is recommended to keep on)
z_RNN_TRAIN         	= 1
z_RNN_TEST          	= 1
z_RECOVER_RESULT   	= 1
#evaluation
z_EVAL_RESULT	   	= 1
z_QUICK_REPORT  	= 1
##############################
#train cfg parameters
max_epochs_no_best = 10
max_epochs = 1000
parallel_sequences = 40
learning_rate = "5e-3"
momentum = 0
#additional option
inputfeat_dim = 3
inputwe_dim = 50
gpu_deviceid = -1
##############################
#Settled variables (need not be changed)
#train
v_ROOT_DIR="exp/lstmnerwer.iii.t1"
preptrain_dir=v_ROOT_DIR+"/traindata"
train_data=preptrain_dir+"/train.nc"
train_log=ROOT_DIR+"/currennt.train.log"
#val
prepval_dir=v_ROOT_DIR+"/valdata"
val_data=prepval_dir+"/val.nc"
#test
preptest_dir=v_ROOT_DIR+"/testdata"
test_data=preptest_dir+"/test.nc"
test_log=ROOT_DIR+"/currennt.test.log"
#model
model_dir=ROOT_DIR+"/model"
#save_network=model_dir+"/autosave/network_train.jsn.epoch012.autosave"
#save_weweights=model_dir+"/autosave/network_train.jsn.we.epoch012.autosave"
#save_featweights=model_dir+"/autosave/network_train.jsn.featweights.epoch012.autosave"
save_network=model_dir+"/network_train.jsn"
save_weweights=model_dir+"/network_train.jsn.we"
save_featweights=model_dir+"/network_train.jsn.featweights"
autosave_prefix=model_dir+"/autosave/network_train.jsn."
#result
result_dir=ROOT_DIR+"/result"
predict_output=result_dir+"/test.predict"
recover_output=result_dir+"/test.recover"
evalresult_file=result_dir+"/evalresult.txt"
quick_report= ROOT_DIR+"/quick_report.txt"
#config
rnntrain_cfg=ROOT_DIR+"/currennt.train.cfg"
rnntest_cfg=ROOT_DIR+"/currennt.test.cfg"
#addional option
vocab_size = 0
##############################
#Main script
def main():
	if(z_PREP_DATA): #generate netCDF data for currennt
		print "\n>> Generating NN data\n"
		print "train data:"
		cmd="python "+SCRIPT_DIR+"/preptraindata_v12.lw_cf3.py "+CONFIG_LOG+" train"
		execmd(cmd)
		print "val data:"
		cmd="python "+SCRIPT_DIR+"/preptraindata_v12.lw_cf3.py "+CONFIG_LOG+" val"
		execmd(cmd)
		print "test data:"
		cmd="python "+SCRIPT_DIR+"/preptraindata_v12.lw_cf3.py "+CONFIG_LOG+" test"
		execmd(cmd)
	if(z_PREP_CFG):
		print "\n>> Prep config...\n"
		cmd="python "+SCRIPT_DIR+"/gen_currenntcfg.py "+CONFIG_LOG+" train"
		execmd(cmd)
		cmd="python "+SCRIPT_DIR+"/gen_currenntcfg.py "+CONFIG_LOG+" test"
		execmd(cmd)
	if(z_RNN_TRAIN):
		print "\n>> Training RNN model...\n"
		cmd=RNN_TOOL+" %s > %s"%(rnntrain_cfg, train_log)
		checkgpu.waitforavailgpu(gpu_deviceid, 60)
		try:
			execmd(cmd)
		except Exception as e:
			print "\nFailed!"
			exit(1)
	if(z_RNN_TEST):
		print "\n>> Testing RNN model...\n"
		cmd=RNN_TOOL+" %s > %s"%(rnntest_cfg, test_log)
		checkgpu.waitforavailgpu(gpu_deviceid, 60)
		try:
			execmd(cmd)
		except Exception as e:
			print "\nFailed!"
			exit(1)
	if(z_RECOVER_RESULT):
		print "\n>> Recorver result...\n"
		cmd="python "+SCRIPT_DIR+"/recover_result.iobes.viterbi.py "+CONFIG_LOG
		execmd(cmd)
	if(z_EVAL_RESULT):
		print "\n>> Eval result...\n"
		cmd="python "+SCRIPT_DIR+"/eval_result.py "+CONFIG_LOG
		execmd(cmd)
		cmd="perl "+SCRIPT_DIR+"/conlleval.pl < "+recover_output+ " > "+evalresult_file
		execmd(cmd)
	if(z_QUICK_REPORT):
		print "\n>> Generate quick report...\n"
		gen_quickreport()

def execmd(cmd):
	print "......"+cmd
	cmd=cmd.replace("/","\\")
	i=os.system(cmd)
	if(i!=0):
		raise Exception,"%s failed"%cmd

def initworkspace(workspace):
	initdir(workspace)
	initdir("%s/model"%workspace)
	initdir("%s/model/autosave"%workspace)
	initdir("%s/traindata"%workspace)
	initdir("%s/valdata"%workspace)
	initdir("%s/testdata"%workspace)
	initdir("%s/result"%workspace)
	initdir("%s/tmp"%workspace)

def initdir(dirpath):
	if(os.path.exists(dirpath)):
		return
	os.mkdir(dirpath)

def store_config(filepath):
	fo=open(filepath,"w")
	gvars=sorted(globals().items(),key=lambda x:x[0],reverse=False)
	fo.write("######################\n")
	fo.write("#config\n")
	fo.write("######################\n")
	fo.write("cwd = %s\n"%(os.getcwd()))
	for var in gvars:
		if(var[0].startswith("__")):
			continue
		if(type(var[1])==types.StringType or type(var[1])==types.IntType or type(var[1])==types.FloatType):
			fo.write(var[0]+" = "+str(var[1])+"\n")
	fo.close()

def getdictsize(dictfile):
	n=0
	fi=open(dictfile)
	for line in fi:
		n+=1
	fi.close()
	return n

def gen_quickreport():
	#get train_log duration, epoches, test error
	fi=open(train_log)
	f=0
	total_dur=0
	total_epoches=0
	best_epoch=0
	best_testerr=0
	for line in fi:
		line=line.strip()
		if(line.startswith("Epoch | Duration")):
			f=1
			continue
		if(f>=1):
			f+=1
		if(f>=3):
			if(line==""):
				break
			toks=line.split("|")
			dur=float(toks[1].strip())
			epoch=toks[0].strip()
			total_dur+=dur
			total_epoches+=1
			bestflag=toks[-1].strip()
			testerr_str=toks[-2].strip()
			ts=testerr_str.split(" ")
			testerr=ts[0].strip("%")
			if(bestflag=="yes"):
				best_epoch=int(epoch)
				best_testerr=float(testerr)
	fi.close()
	best_accuracy=100-best_testerr
	avg_epoch_dur=float(total_dur)/total_epoches
	#get eval result
	eval_errorrate=0
	evalresult_lines=[]
	fi=open(evalresult_file)
	i=0
	f1score=0
	for line in fi:
		i+=1
		line=line.strip()
		evalresult_lines.append(line)
		if(i==2):
			toks=line.split(";")
			for tok in toks:
				tok=tok.strip()
				if(tok.startswith("FB1")):
					f1score=float(tok[4:].strip())
	#generate report
	fo=open(quick_report,"w")
	fo.write("epoch dur || epoches || accuracy || f1 ||\n")
	fo.write("%d"%avg_epoch_dur+" || %d"%best_epoch+" || %.2f"%best_accuracy+" || %.2f"%f1score+" ||\n\n")
	for line in evalresult_lines:
		fo.write(line+"\n")
	fo.close()

if __name__ == '__main__':
	try:
		#init workspace, make dir
		print "\n>> Initiating workspace: %s\n"%ROOT_DIR
		initworkspace(ROOT_DIR)
		vocab_size=getdictsize(WORD_DICT)
		store_config(CONFIG_LOG)
		main()
	except Exception as e:
		print "!!!!!! "+__file__+" exit with exception !!!!!!"
		print e
