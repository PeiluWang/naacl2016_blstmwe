#!/usr/bin/python
#coding=utf-8
"""
read lstm layer's weights from json file
"""
import json

fi=open("network_train.jsn","r")
data=json.load(fi)
fi.close()

lstm_section=data["weights"]["hidden_layer"]
lstm_input=lstm_section["input"]
lstm_bias=lstm_section["bias"]
lstm_internal=lstm_section["internal"]

total_size=len(lstm_input)+len(lstm_bias)+len(lstm_internal)
print "total weights: %d"%(total_size)

fo=open("lstm_weights.txt","w")
for w in lstm_input:
	fo.write("%e"%w+"\n")
for w in lstm_bias:
	fo.write("%e"%w+"\n")
for w in lstm_internal:
	fo.write("%e"%w+"\n")
fo.close()

