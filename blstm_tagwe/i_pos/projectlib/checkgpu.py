"""
check if gpu is available
"""
import os
import time
import random

#check if gpu is available, if not, wait
def waitforavailgpu(gpu_deviceid, check_gap):
	print "... wait for available gpu"
	while(True):
		available_gpus=update_gpustat()
		run=False
		if(len(available_gpus)>0):
			if(gpu_deviceid>=0):
				if(gpu_deviceid in available_gpus):
					run=True
			else:
				run=True
		if(run):
			print "... detect available gpu: "+str(list(available_gpus))
			break
		gap=random.randint(0, check_gap/2)+check_gap
		time.sleep(gap)
	

def update_gpustat():
	gpu_stat=[0,0]
	outputs=os.popen("nvidia-smi").readlines()
	f=0
	for line in outputs:
		if(line.startswith("|  GPU       PID")):
			f=1
			continue
		if(f==1):
			f=2
			continue
		if(f==2):
			if(not line.startswith("|")):
				break
			id=-1
			for c in line[1:]:
				if(c==" "):
					continue
				if(c=="N"): #No running ...
					break
				id=int(c)
				break
			if(id==-1):
				break
			gpu_stat[id]=1
	available_gpus=set()
	for i in range(len(gpu_stat)):
		if(gpu_stat[i]==0):
			available_gpus.add(i)
	return available_gpus