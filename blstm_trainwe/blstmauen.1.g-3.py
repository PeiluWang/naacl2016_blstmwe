#!/usr/bin/python
#coding=utf-8
import os
import types
import sys
import traceback
import json
if("toollib" not in sys.path):
	sys.path.append("toollib")
import checkgpu
"""
BLSTM RNN for training word embedding
"""
##############################
#Configuration
EXP_NAME=__file__.rstrip(".py")
ROOT_DIR="exp/%s"%EXP_NAME
#data
TRAIN_DATA="../data/rawtxt/ldc/ldc_train.all.f"
VAL_DATA="../data/rawtxt/ldc/ldc_dev.f"
TEST_DATA="../data/rawtxt/conll03ner/conll03ner.test.f"
#config file
NETWORK_CFG="resource/network.i50b100o2.jsn"
TRAIN_CFG_TEMPLATE="resource/exp.train.template.cfg"
TEST_CFG_TEMPLATE="resource/exp.test.template.cfg"
WORD_DICT="data/lowerworddict.ldcallf-cwf_130k_50dim"
load_weweights = "data/weweights/weweights.ldcallf-cwf_130k_50dim.cwf_130k_50dim"
load_featweights = "none"
load_lstmweights = "none"
#tool
SCRIPT_DIR="script"
RNN_TOOL="../tool_rnn/currennt_v12d/currennt/Release/currennt.exe"
#log
CONFIG_LOG=ROOT_DIR+"/config.log"
#############################
#Step Switches
#prepare data
z_PREP_DATA		= 1
#train and test
z_PREP_CFG              	= 1     #  Generate config file (this switch is recommended to keep on)
z_RNN_TRAIN         	= 1
z_GEN_WEDICT 	= 1
##############################
#train cfg parameters
max_epochs_no_best = 3
max_epochs = 1000
parallel_sequences = 40
learning_rate = "5e-3"
momentum = 0
replacerate = 0.2
#additional option
inputfeat_dim = 1
inputwe_dim = 50
gpu_deviceid = -1
#parallel
prep_fraction_size = 1147900
prep_fraction_startid = 0
prep_fraction_endid = 20
thread_num = 6
##############################
#Settled variables (need not be changed)
#train
v_ROOT_DIR=ROOT_DIR
preptrain_dir=v_ROOT_DIR+"/traindata"
train_data=preptrain_dir+"/train.nc."+str(prep_fraction_size)
#val
prepval_dir=v_ROOT_DIR+"/valdata"
val_data=prepval_dir+"/val.nc"
#test
preptest_dir=v_ROOT_DIR+"/testdata"
test_data=preptest_dir+"/test.nc"
#model
model_dir=ROOT_DIR+"/model"
save_network=model_dir+"/network_train.jsn"
save_weweights=model_dir+"/network_train.jsn.we"
save_featweights=model_dir+"/network_train.jsn.featweights"
autosave_prefix=model_dir+"/autosave/network_train.jsn."
#generate wedict
result_dir=ROOT_DIR+"/result"
trained_we=model_dir+"/autosave/network_train.jsn.epoch010.autosave.we"
wedict=result_dir+"/we."+EXP_NAME.replace(".","_")+".epoch10.%ddim"%inputwe_dim
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
		cmd="python "+SCRIPT_DIR+"/preptraindata.lw_r.parallel.py "+CONFIG_LOG+" train"
		execmd(cmd)
		print "val data:"
		cmd="python "+SCRIPT_DIR+"/preptraindata.lw_r.parallel.py "+CONFIG_LOG+" val"
		execmd(cmd)
		print "test data:"
		cmd="python "+SCRIPT_DIR+"/preptraindata.lw_r.parallel.py "+CONFIG_LOG+" test"
		execmd(cmd)
	if(z_PREP_CFG):
		print "\n>> Prep config...\n"
		cmd="python "+SCRIPT_DIR+"/gen_currenntcfg.py "+CONFIG_LOG+" train"
		execmd(cmd)
		cmd="python "+SCRIPT_DIR+"/gen_currenntcfg.py "+CONFIG_LOG+" test"
		execmd(cmd)
	if(z_RNN_TRAIN):
		print "\n>> Training RNN model...\n"
		cmd=RNN_TOOL+" %s > %s/currennt.train.log"%(rnntrain_cfg,ROOT_DIR)
		checkgpu.waitforavailgpu(gpu_deviceid, 60)
		try:
			execmd(cmd)
		except Exception as e:
			print "\nFailed!"
			exit(1)
	if(z_GEN_WEDICT):
		print "\n>> Generating WE dict...\n"
		cmd="python "+SCRIPT_DIR+"/gen_wedict.py "+CONFIG_LOG
		execmd(cmd)

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

def get_datadim():
	fi=open(NETWORK_CFG,"r")
	content=fi.read()
	fi.close()
	networkjsn=json.loads(content)
	inputdim=networkjsn["layers"][0]["size"]
	outputdim=networkjsn["layers"][-1]["size"]
	return inputdim,outputdim

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

if __name__ == '__main__':
	try:
		#init parameters
		input_dim,output_dim=get_datadim()
		#init workspace, make dir
		print "\n>> Initiating workspace: %s\n"%ROOT_DIR
		initworkspace(ROOT_DIR)
		vocab_size=getdictsize(WORD_DICT)
		store_config(CONFIG_LOG)
		main()
	except Exception as e:
		print "!!!!!! "+__file__+" exit with exception !!!!!!"
		print e

