#!/usr/bin/python
#coding=utf-8
"""
Prepare the RNN training data:
Convert the F0 and LSP to RNN output
"""
import sys
import traceback

CONFIG={}

def main():
	fi=open(CFG_TEMPLATE)
	fo=open(CFG_FINAL,"w")
	configs=[]
	for line in fi:
		if(line.startswith("#")):
			continue
		line=line.strip()
		if(line==""):
			continue
		toks=line.split("=")
		var=toks[0].strip()
		value=toks[1].strip()
		value=value.replace("$ROOT_DIR",ROOT_DIR)
		value=value.replace("$train_data",train_data)
		value=value.replace("$val_data",val_data)
		value=value.replace("$test_data",test_data)
		value=value.replace("$NETWORK_CFG",NETWORK_CFG)
		value=value.replace("$save_network",save_network)
		value=value.replace("$save_weweights",save_weweights)
		value=value.replace("$load_weweights",load_weweights)
		value=value.replace("$save_featweights",save_featweights)
		value=value.replace("$load_featweights",load_featweights)
		value=value.replace("$load_lstmweights",load_lstmweights)
		value=value.replace("$autosave_prefix",autosave_prefix)
		value=value.replace("$predict_output",predict_output)
		value=value.replace("$max_epochs_no_best",max_epochs_no_best)
		value=value.replace("$max_epochs",max_epochs)
		value=value.replace("$learning_rate",learning_rate)
		value=value.replace("$parallel_sequences",parallel_sequences)
		value=value.replace("$momentum",momentum)
		value=value.replace("$vocab_size",vocab_size)
		value=value.replace("$vocab_class",vocab_class)
		value=value.replace("$inputwe_dim",inputwe_dim)
		value=value.replace("$inputfeat_dim",inputfeat_dim)
		value=value.replace("$gpu_deviceid",gpu_deviceid)
		value=value.replace("/","\\")
		configs.append((var,value))
	configs=sorted(configs,key=lambda x:x[0],reverse=False)
	for config in configs:
		fo.write("%s = %s\n"%(config[0],config[1]))
	fi.close()
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

if __name__ == '__main__':
	if(len(sys.argv)!=3):
		print >> sys.stderr, "USAGE: %s configlog"%(__file__)
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),3)
		exit(1)
	try:
		configlog=sys.argv[1]
		type=sys.argv[2]
		loadconfig()

		if(type=="train"):
			CFG_TEMPLATE=CONFIG["TRAIN_CFG_TEMPLATE"]
			CFG_FINAL=CONFIG["rnntrain_cfg"]
		elif(type=="test"):
			CFG_TEMPLATE=CONFIG["TEST_CFG_TEMPLATE"]
			CFG_FINAL=CONFIG["rnntest_cfg"]
		else:
			raise Exception, "type is not train, val or test: "+type
		ROOT_DIR=get_configvalue("ROOT_DIR")
		NETWORK_CFG=get_configvalue("NETWORK_CFG")
		train_data=get_configvalue("train_data")
		val_data=get_configvalue("val_data")
		test_data=get_configvalue("test_data")
		predict_output=get_configvalue("predict_output")
		max_epochs_no_best = get_configvalue("max_epochs_no_best")
		max_epochs = get_configvalue("max_epochs")
		learning_rate = get_configvalue("learning_rate")
		parallel_sequences = get_configvalue("parallel_sequences")
		momentum = get_configvalue("momentum")
		vocab_size = get_configvalue("vocab_size")
		vocab_class = get_configvalue("vocab_class")
		gpu_deviceid = get_configvalue("gpu_deviceid")
		save_network = get_configvalue("save_network")
		save_weweights = get_configvalue("save_weweights")
		load_weweights = get_configvalue("load_weweights")
		save_featweights = get_configvalue("save_featweights")
		load_featweights = get_configvalue("load_featweights")
		load_lstmweights = get_configvalue("load_lstmweights")
		autosave_prefix = get_configvalue("autosave_prefix")
		inputwe_dim = get_configvalue("inputwe_dim")
		inputfeat_dim = get_configvalue("inputfeat_dim")
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)
