# build.py
import os
import pickle
import numpy as np
from html.parser import HTMLParser
from sklearn.feature_extraction.text import TfidfVectorizer

# Concatenates the content of all <p> for each <doc> of the fed data, also saves <doc_no> 
class docParser(HTMLParser):
    docs_no = []

    def __init__(self):
        super().__init__()
        self.docs = [] 
        self.tagPile = ["root"] # root to avoid index out of bound when using tagPile[-1]

    def handle_starttag(self, tag, attrs):
        self.tagPile.append(tag)
        if self.tagPile[-1] == "doc":
            self.docs.append("")

    def handle_endtag(self, tag):
        self.tagPile.pop()

    def handle_data(self, data):
        if self.tagPile[-1] == "p":
            self.docs[-1]+=data
        elif self.tagPile[-1] == "docno":
            docParser.docs_no.append(data)

# Takes a file and returns a list of docs
def parseDocs(f):
    d = docParser()
    d.feed(f.read())
    return d.docs

# Reads up to lim files (or all if lim<0) in alphabetical order from dataDir and returns a list of docs 
def readFiles(dataDir, lim = -1):
    train_set = []
    for file in np.sort(os.listdir(dataDir)):
        with open(os.path.join(dataDir, file), 'r') as f:
            train_set.extend(parseDocs(f))			
        lim = lim - 1
        if(lim==0):	break

    with open('docs_no.pkl', 'wb') as f:
        pickle.dump(docParser.docs_no, f)
    return train_set

# load data, tokenize and build voc
dataset = readFiles('./latimes', -1) # do not forget to remove the 2 .txt instruction files from the dataset !
vectorizer = TfidfVectorizer(dtype=np.float32, stop_words='english', use_idf=True, max_df=0.5) # utiliser float16 déclenche une exception
idfTable = vectorizer.fit_transform(dataset).tocsc() # the voc contains roughly 250,000 words 5% of which are numbers
with open('voc.pkl', 'wb') as f:
    pickle.dump(vectorizer.vocabulary_, f)

# Save the inverted file and its corresponding index (byte offset for each word's doc_scores)
# inv_file format: <nb_docs:uint16><docs:[uint32]><scores:[float16]> ... 
word_count = len(idfTable.indptr)-1
posWords = np.ndarray(word_count, dtype=np.uint32)
with open("inv_file.bin", "wb") as bin_file:
	for i in range(word_count):
		start = idfTable.indptr[i]
		end = idfTable.indptr[i+1]
		
		docs = np.array(idfTable.indices[start:end], dtype=np.uint32)
		scores = np.array(idfTable.data[start:end], dtype=np.float16)
		
		arg_s = np.argsort(scores)[::-1]
		docs = docs[arg_s]
		scores = scores[arg_s]
		
		posWords[i] = bin_file.tell()
		bin_file.write(np.uint16(len(docs)).tostring() + docs.tostring() + scores.tostring())
posWords.tofile("inv_file_index.bin")