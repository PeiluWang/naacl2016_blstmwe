fi=open("train.txt","r")
toknum=0
wordset=set()
lowerwordset=set()
posset=set()
tagdict=dict()
sennum=0
for line in fi:
	line=line.strip()
	if(line==""):
		sennum+=1
		continue
	toks=line.split(" ")
	toknum+=1
	wordset.add(toks[0])
	lowerwordset.add(toks[0].lower())
	posset.add(toks[1])
	if(toks[2] not in tagdict):
		tagdict[toks[2]]=0
	tagdict[toks[2]]+=1

print "sen num: %d"%sennum
print "token num: %d"%toknum
print "wordset num: %d"%len(wordset)
print "lowerwordset num: %d"%len(lowerwordset)
print "posset num: %d"%len(posset)
print "tagset num: %d"%len(tagdict)

fo=open("taglist.txt","w")
taglist=sorted(tagdict.iteritems(), key=lambda x:x[0], reverse=True)
for item in taglist:
	fo.write(item[0]+"\t"+str(item[1])+"\n")
fo.close()
