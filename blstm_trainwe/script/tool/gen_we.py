words=[]
fi=open("../../../resource/ldc.10m.worddict100k.txt","r")
for line in fi:
	line=line.strip()
	words.append(line)
fi.close()

l=len(words)
fi=open("network_train.jsn.we","r")
fo=open("we.ldc10m.100k.200dim.txt","w")
i=0
for line in fi:
	line=line.strip()
	toks=line.split(" ")
	fo.write(words[i]+"\t"+"\t".join(toks)+"\n")
	i+=1
fi.close()
fo.close()
assert i==l

print "done"

