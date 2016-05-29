#generate word dict
#contain all common words of lowerworddict.nerrf
#contain the most frequenct word of the other dict
#total 100k words

nerrfdict_fp="../../tagging_ner/data/lowerworddict.nerrf.txt"
indict_fp="../../data/rawtxt/ldc/ldc_train.100m.f.lowercasewordcount"
outdict_fp="lowerworddict.nerrf-ldc100mf.100k"
nerrf_wordset=set()
common_wordset=set()
#load nerrf wordset
fi=open(nerrfdict_fp,"r")
for line in fi:
	line=line.strip()
	nerrf_wordset.add(line)
fi.close()
#load indict
inwords=[]
fi=open(indict_fp,"r")
for line in fi:
	toks=line.split("\t")
	word=toks[0]
	inwords.append(word)
	if(word in nerrf_wordset):
		common_wordset.add(word)
fi.close()
common_wc=len(common_wordset)
final_wc=common_wc
fo=open(outdict_fp,"w")
if("<unk>" not in common_wordset):
	fo.write("<unk>"+"\n")
	final_wc+=1
for word in common_wordset:
	fo.write(word+"\n")
#output first 100k words
for word in inwords:
	if(word not in common_wordset):
		fo.write(word+"\n")
		final_wc+=1
		if(final_wc>=100000):
			break
fo.close()

nerrf_wc=len(nerrf_wordset)
print "nerrf wc: %d"%nerrf_wc
print "common wc: %d"%common_wc
print "nerrf oov: %f"%((float(nerrf_wc)-common_wc)/nerrf_wc)

