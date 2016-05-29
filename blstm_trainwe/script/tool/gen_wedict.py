worddict_file="../../../../data/we/lowerworddict.wsj_tf.common-ldcallf.txt"
words=[]
fi=open(worddict_file,"r")
for line in fi:
	line=line.strip()
	words.append(line)
fi.close()

for i in range(1,3,1):
	print i
	we_file="network_train.jsn.epoch%03d.we.autosave"%i
	fo=open("we.ldcallf.wsjtfc-ldcallf_100k_100dim.%d"%i,"w")
	fi=open(we_file,"r")
	i=0
	for line in fi:
		line=line.strip()
		toks=line.split(" ")
		fo.write(words[i]+"\t"+"\t".join(toks)+"\n")
		i+=1
	fi.close()
	fo.close()
	assert i==len(words)

