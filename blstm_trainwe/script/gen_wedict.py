import sys
import traceback

def main():
	words=[]
	fi=open(WORD_DICT,"r")
	for line in fi:
		line=line.strip()
		words.append(line)
	fi.close()

	fo=open(wedict,"w")
	fi=open(trained_we,"r")
	i=0
	for line in fi:
		line=line.strip()
		toks=line.split(" ")
		fo.write(words[i]+"\t"+"\t".join(toks)+"\n")
		i+=1
	fi.close()
	fo.close()
	assert i==len(words)

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
		print >> sys.stderr, "Your args' number is %d != %d"%(len(sys.argv),3)
		exit(1)
	try:
		CONFIG={}
		configlog=sys.argv[1]
		loadconfig()
		#load parameters
		WORD_DICT=CONFIG["WORD_DICT"]
		trained_we=CONFIG["trained_we"]
		wedict=CONFIG["wedict"]
		main()
	except Exception as e:
		print >> sys.stderr, str(e)
		print >> sys.stderr, traceback.format_exc()
		exit(1)