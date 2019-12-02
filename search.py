# search.py
import numpy as np
from collections import defaultdict
import heapq
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timeit

from scipy import stats

with open('docs_no.pkl', 'rb') as do, open('voc.pkl', 'rb') as vo:
    docs_no = pickle.load(do)
    voc = pickle.load(vo)
index = np.fromfile("inv_file_index.bin", dtype=np.uint32)

# For a given word, returns a dict {doc1: score1, ...} ordered by decreasing score
def loadDocScores(word):
	with open("inv_file.bin", "rb") as bin_file:
		bin_file.seek(index[voc[word]])
		size = np.frombuffer(bin_file.read(2), dtype=np.uint16)[0]
		doc = np.frombuffer(bin_file.read(size*4), dtype=np.uint32)
		score = np.frombuffer(bin_file.read(size*2), dtype=np.float16)
	return defaultdict(np.float16, zip(doc,score)) # from python 3.6 dicts are sorted by insertion order, made official in 3.7

# Naive algo returning the k most relevant documents for a given query (OoV words are ignored)
def naive(query, k):
	c = defaultdict(np.float16)
	ds = [loadDocScores(word=qt) for qt in query.lower().split(" ") if qt in voc]
	for qt in ds:
		for doc, score in qt.items():
			c[doc]+=score
	return heapq.nlargest(k, c, key=c.get)

# Fagin's algo returning the k most relevant documents for a given query (OoV words are ignored)
# We manage the case where one or all query terms run out of documents before we have k docs in c by exiting the loop
def fagin(query, k):
	m = {}
	c = defaultdict(np.float16)
	ds = [loadDocScores(word=qt) for qt in query.lower().split(" ") if qt in voc]
	its = [iter(dic.items()) for dic in ds]
	its_end = list(range(len(its)))
	while (len(c) < k) and its_end: # exit the loop when no iterators are left
		for j in its_end:
			try:
				doc, score = next(its[j])
			except StopIteration: 
				its_end.remove(j)
				if not its_end: break # exit the loop when no iterators are left
				continue
			if doc in m:
				m[doc][0].append(score)
				m[doc][1].remove(j)
			else:
				m[doc] = ([score], [x for x in range(len(ds)) if x!=j]) # (score, qt to check)
			if not m[doc][1]: # doc seen for all qt
				c[doc] = sum(m[doc][0])
				del m[doc]

	for doc, v in m.items():
		if its_end:
			for j in v[1]: # list of qt to check
				v[0].append(ds[j][doc]) # if doc is not in the dict it adds it with score 0.0
		c[doc] = sum(v[0]) # sum of scores
	return heapq.nlargest(k, c, key=c.get)

#### Example use
def example(q="the little green turtle of the zoo", k=5):

	rn = naive(q,k)
	rf = fagin(q,k)

	docs_rn = [docs_no[i] for i in rn]
	docs_rf = [docs_no[i] for i in rf]

	print("rn: ", rn, " docs_naive: ", docs_rn)
	print("rf: ", rf, " docs_fagin: ", docs_rf)

#### Tests
exp = 3
top_k = [x*10**y for y in range(exp) for x in range(1,10)]
top_k.append(10**exp)

qs = [
		"the little green turtle",
		"i like chocolate",
		"workers car factory",
		"birds beginning spring",
		"autumn leaves fall",
		"how to plot graphs in python",
		"Unusual Smell Coming From Sewer",
		"Missing Dog Turns Up After 2 Years",
		"'Chatty' Monkey Escapes From Zoo",
		"Earthquake - Town In Ruins",
		"Helicopter Lands In School Playground",
		"Girl, 10, Saves Friend With Rubber Ring",
		"Eminem Terrified As Daughter Begins Dating Man Raised On His Music",
		"7 Ways to Make Money While Waiting for Disability Benefits",
		"How to Have a Healthier and More Productive Home Office",
		"A Little Mistake That Cost a Farmer $3,000 a Year",
		"Are You Making These Embarrassing Mistakes at Work?",
		"Lose 8 Pounds in 2 Weeks",
		"How Many of These Italian Foods Have You Tried?",
		"What’s Scarier Than the Sex Talk? Talking About Food & Weight",
		"More Than Half of Medical Advice on ‘Dr. Oz’ Lacks Proof or Contradicts Best Available Science",
		"Lack Time? Here Are 4 Convenient Ways to Keep Your Dog Fit",
		"How One Stupid Tweet Blew Up Justine Sacco’s Life",
		"10 Signs That You Will NOT Make It As A Successful Photographer",
		"Sure-Fire Ways to Ruin Your Marriage",
		"10 Different Types of Girlfriends – Which One Are You?",
		"More of Us May Be “Almost Alcoholics”",
		"Content Killers: Headlines That Never Pan Out",
		"Make One Million Dollars in One Day",
		"Study Shows Frequent Sex Enhances Pregnancy Chances",
		"We Didn’t Believe It. So We Fact-Checked It (Twice). Now Let’s Talk About How to Take It Worldwide.",
		"“Star Wars: The Force Awakens” Ultimate Guide",
		"Health Insurance Companies HATE This New Trick",
		"The Weight Loss Trick That Everyone Is Talking About",
		"This Stick Of Butter Is Left Out At Room Temperature; You Won’t Believe What Happens Next",
		"The DOJ Just Released Its Ferguson Investigation — And What They Found Was Horrifying",
		"60 Photos From the Past That Will Blow Your Mind",
		"What state has highest rate of rape in the country? It may surprise you.",
		"5 Reasons To Date A Girl With An Eating Disorder",
		"Weight Loss Shakes Lose Weight Today",
		"No Results With Your Attempts to Stop Drinking?",
		"The Importance of the Legal Aspects of Business Correspondence",
		"How to Tie Your Shoes",
		"Dead Body Found in Cemetery",
		"One-Armed Man Applauds the Kindness of Strangers",
		"Infusion Partners With Anheuser-Busch to Accelerate Business Innovation Using Microsoft Hololens",
		"How To Write Award Winning Blog Headlines",
		"No, Spooning Isn’t Sexist. The Internet Is Just Broken.",
		"Ebola in the air? A nightmare that could happen.",
		"These Workers Just Want Money, And You Won’t Believe What They Did To Get Some.",
		"Someone Gave Some Kids Some Scissors. Here’s What Happened Next."
	]

# Affiche des statistiques sur la vitesse des deux algorithmes
# appliqués à une liste de requêtes pour différentes tailles de top k
def testSpeed():

	size = (len(qs), len(top_k))
	rn = np.ndarray(size, dtype=np.float32)
	rf = np.ndarray(size, dtype=np.float32)

	rep = 5
	for i, q in enumerate(qs):
		for j, k in enumerate(top_k):
			rn[i][j] = timeit.timeit(lambda: naive(q, k), number=rep)/rep
			rf[i][j] = timeit.timeit(lambda: fagin(q, k), number=rep)/rep

	rn_stats = stats.describe(rn)
	rf_stats = stats.describe(rf)

	mk=6

	fig, axs = plt.subplots(nrows=2, ncols=1)
	axs[0].plot(top_k, rn_stats.mean, "bo-", markersize=mk, label="mean")
	axs[1].plot(top_k, rf_stats.mean, "bo-", markersize=mk, label="mean")

	axs[0].plot(top_k, rn_stats.minmax[0], "gv-", markersize=mk, label="min")
	axs[1].plot(top_k, rf_stats.minmax[0], "gv-", markersize=mk, label="min")

	axs[0].plot(top_k, rn_stats.minmax[1], "g^-", markersize=mk, label="max")
	axs[1].plot(top_k, rf_stats.minmax[1], "g^-", markersize=mk, label="max")

	axs[0].errorbar(top_k, rn_stats.mean, np.sqrt(rn_stats.variance), fmt="b", markersize=mk, label="std")
	axs[1].errorbar(top_k, rf_stats.mean, np.sqrt(rf_stats.variance), fmt="b", markersize=mk, label="std")

	axs[0].set_xscale("log")
	axs[0].set(xlabel='top k (#docs)', ylabel='time (s)',
	       title='Time of naive search depending on #docs in top')

	axs[1].set_xscale("log")
	axs[1].set(xlabel='top k (#docs)', ylabel='time (s)',
	       title='Time of fagin''s search depending on #docs in top')

	axs[0].grid()
	axs[1].grid()
	plt.show()
	#fig.savefig("test_speed.png")

# Affiche l'histogramme du pourcentage d'intersection des deux ensembles de documents obtenus (intersect)
# et l'histogramme du pourcentage d'éléments à la même place dans les deux ensembles (overlap)
def testSimilarity():
	size = (len(qs), len(top_k))
	intersect = np.ndarray(size, dtype=np.float32)
	overlap = np.ndarray(size, dtype=np.float32)

	for i, q in enumerate(qs):
		for j, k in enumerate(top_k):
			rn = naive(q,k)
			rf = fagin(q,k)
			intersect[i][j] = len(np.intersect1d(rn, rf, assume_unique=True))/k
			overlap[i][j] = sum(np.equal(rn, rf))/k 
	intersect = np.mean(intersect, axis=0)
	overlap = np.mean(overlap, axis=0)

	df = pd.DataFrame({'intersect': intersect, 'overlap': overlap}, index=top_k)
	ax = df.plot.bar()	
	ax.set(xlabel='top k (#docs)', ylabel='similarity',
	title='Similarity of top k docs depending on #docs')
	plt.show()
	#fig.savefig("test_similarity.png")