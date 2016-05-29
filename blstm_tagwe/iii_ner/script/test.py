
word="HELL1"

wordl=word.lower()
if(word==wordl):
	hasuppercase=0
else:
	hasuppercase=1
wordu=word.upper()
if(word==wordu):
	allisupper=1
else:
	allisupper=0
w1=word[0]
w1u=w1.upper()
if(w1==w1u):
	firstisupper=1
else:
	firstisupper=0

print hasuppercase
print allisupper
print firstisupper