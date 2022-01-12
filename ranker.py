import math
import re
import os
import ast

class Ranker:
    def __init__(self, queries_file, tokenizer, method="vector", docs_limit=100, k1=1.2, b=0.75, doc_lengths=None, avg_dl=None):
        self.queries_file = queries_file
        self.tokenizer = tokenizer
        self.method = method
        self.docs_limit = docs_limit
        self.k1 = k1
        self.b = b
        self.temp_index = {}
        self.doc_lengths = doc_lengths
        self.avg_dl = avg_dl
        self.queries_results = {}
        self.dictionary = {}


    def run(self):
        #Process queries
        queries_list = self.readQuery()
        self.dictionary = self.readDictionary()
        for i in range(len(queries_list)):
            indexed_query = self.query_freq(self.tokenizer.get_tokens(queries_list[i], i, False))

            print("Searching {}º query...".format(i+1))
            best_docs = {}
            if self.method == 'vector':
                best_docs = self.rank_vector(indexed_query)
            elif self.method == 'bm25':
                best_docs = self.rank_bm25(indexed_query)

            self.queries_results[queries_list[i]] = best_docs

            #testing only first query 
            #break

    def readDictionary(self):
        f = open("dictionary/dictionary.txt")
        return ast.literal_eval(f.read())

    def readQuery(self):
        queries_list = []
        with open(self.queries_file) as f:
            lines = f.readlines()
            if lines:
                queries_list = [line.rstrip() for line in lines]
        
        return queries_list


    def writeResults(self):
        query_file = self.queries_file.split("/")[-1]
        with open("search_output/results_"+ str(query_file) + "_" + str(self.method)+".txt",'w') as f:
            for query, doc_list in self.queries_results.items():
                f.write("Q: {}\n".format(query))
                for doc in doc_list:
                    f.write(str(doc[0])+"\n")
                f.write("\n")


    def query_freq(self,tokens):
        indexed_query = {}
        for token in tokens:
            # Desagragate tuple
            term = token[0]

            if term not in indexed_query.keys():
                indexed_query[term] = 1
            else:
                indexed_query[term] += 1
        return indexed_query

    def rank_vector(self, indexed_query):
        scores = {}     #best docs

        query_norm = 0
        docs_norm = {}
        for term, tf in indexed_query.items():
            # Find where term is located
            right_index_file = self.findIndexFile(term)
            # idf for each term    
            idf = self.dictionary[term][0]
            
            tf_weight = math.log10(tf) + 1
            weight_query_term = tf_weight * float(idf) # Weight for the term in the query
            query_norm += weight_query_term ** 2

            for doc_id, doc_weight in self.temp_index[right_index_file][term]['docs'].items():
                # Norm
                if doc_id not in docs_norm:
                    docs_norm[doc_id] = doc_weight ** 2
                else:
                    docs_norm[doc_id] += doc_weight ** 2

                score = (weight_query_term * doc_weight)
                if doc_id not in scores:
                    scores[doc_id] = score
                else:
                    scores[doc_id] += score

        
        # length normalize all scores
        for docID, score in scores.items():
            length = math.sqrt(docs_norm[docID]) *  math.sqrt(query_norm)
            scores[docID] /= length

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return best_docs[:self.docs_limit] 
    

    def rank_bm25(self, indexed_query):
        scores = {}     #best docs
        
        for term, tf in indexed_query.items():
            # Find where term is located
            right_index_file = self.findIndexFile(term)

            # idf for each term
            idf = self.dictionary[term][0]
            
            # weight = tf * idf 
            # logo, tf = weight / idf
            for doc_id, doc_weight in self.temp_index[right_index_file][term]['docs'].items():
                doc_tf = doc_weight / idf
                doc_length = self.doc_lengths[doc_id]
                score = self.calc_bm25(doc_tf, doc_length, self.avg_dl, idf)

                if doc_id not in scores:
                    scores[doc_id] = score
                else:
                    scores[doc_id] += score
        
        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return best_docs[:self.docs_limit]
    
    def calc_bm25(self, doc_tf, doc_length, avg_dl, idf):    
        return idf * (((self.k1 + 1) * doc_tf) / (self.k1 * ((1-self.b) + self.b * doc_length / avg_dl) + doc_tf))
        

    def findIndexFile(self, term):
        indexes_list = [file for file in os.listdir("index") if file != '.DS_Store']
        
        right_index_file = None
        for index_file in indexes_list:
            range = index_file.split('.')[0]
            first_word, last_word = range.split('_')[1:]
            if term >= first_word and term <= last_word:
                right_index_file = range
                with open("index/" + index_file) as f:
                    for line in f.readlines():
                        term_file,value = re.split('; ', line.rstrip('\n'), maxsplit=1)
                        #term_idf = self.dictionary[term][0]
                        if range not in self.temp_index.keys():
                            self.temp_index[range] = {}
                        self.temp_index[range][term_file] = { "docs": ast.literal_eval(value)}
                            
                # we find the right index
                break
        
        return right_index_file