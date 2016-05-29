CoNLL2003
	CoNLL 2003 shared task training, development and testing data
		* For named entity recognition (NER) task

	Url:
	http://www.cnts.ua.ac.be/conll2003/

	Ref:
	Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Language-independent named entity recognition. In Proceedings of CoNLL, pages 142–147, Edmonton, Canada. Association for Computational Linguistics.

	The construction of this corpus also need Reuters corpus: RCV1
		Url:
		http://trec.nist.gov/data/reuters/reuters.html

		Ref:
		Lewis, D. D.; Yang, Y.; Rose, T.; and Li, F. RCV1: A New Benchmark Collection for Text Categorization Research. Journal of Machine Learning Research, 5:361-397, 2004. 

		Available at:
		CoNLL2003/origin/RCVCD1

	Constructed corpus for NER task is available at: CoNLL2003/processed_data
		eng.train
			training data
		eng.testa
			valid data
		eng.testb
			test data

		eng.train_rf
		eng.testa_rf
		eng.testb_rf
			remove non-sense line
			consective digits are converted to #, e.g. tel192 -> tel#
