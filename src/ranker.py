from heapq import heappop, heappush
import imp
import math
import re
import os
import ast
import sys
import time
from node import Node

class Ranker:
    def __init__(self, queries_file, tokenizer, boost_flag, method="vector", docs_limit=50, k1=1.2, b=0.75, doc_lengths=None, avg_dl=None):
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
        self.boost_flag = boost_flag


    def run(self):
        #Process queries
        queries_list = self.readQuery()
        self.dictionary = self.readDictionary()

        # Read queries relevance to evaluate
        query_rel_list = self.readQueryRel()
        for i in range(len(queries_list)):
            indexed_query = self.query_freq(self.tokenizer.get_tokens(queries_list[i], i, False))

            print(indexed_query)
            begin = time.time()
            print("Searching {}º query...".format(i+1))
            best_docs = []
            if self.method == 'vector':
                best_docs = self.rank_vector2(indexed_query)
            elif self.method == 'bm25':
                best_docs = self.rank_bm25(indexed_query)

            # Evaluate query
            end = time.time()
            self.evaluateQuery(query_rel_list[i+1], best_docs, end - begin)

            self.queries_results[queries_list[i]] = best_docs
            
            #testing only first query 
            break

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
        # Check whether path exists or not
        if not os.path.exists("search_output"):
            # Create a new directory because it does not exist
            os.makedirs("search_output")
        query_file = self.queries_file.split("/")[-1]
        with open("search_output/results_"+ str(query_file.split('.')[0]) + "_" + str(self.method)+".txt",'w') as f:
            for query, doc_list in self.queries_results.items():
                f.write("Q: {}\n".format(query))
                for doc in doc_list:
                    f.write(str(doc[0])+"\t"+str(doc[1])+"\n")
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

        query_len = 0   # Norm query
        docs_norm = {}
        test_boost = {}
        for term, tf in indexed_query.items():
            # Find where term is located
            print("--> ",term)
            right_index_file = self.findIndexFile(term)
            print(right_index_file)
            # idf for each term    
            idf = self.dictionary[term][0]
            
            tf_weight = math.log10(tf) + 1
            weight_query_term = tf_weight * float(idf) # Weight for the term in the query
            query_len += weight_query_term ** 2

            for doc_id, doc_weight in self.temp_index[right_index_file][term]['docs'].items():
                    # Boost
                    if self.boost_flag:
                        if doc_id not in test_boost:
                            test_boost[doc_id] = [doc_weight[1]]
                        else:
                            test_boost[doc_id].append(doc_weight[1])

                    # Norms
                    if doc_id not in docs_norm:
                        docs_norm[doc_id] = doc_weight[0] ** 2
                    else:
                        docs_norm[doc_id] += doc_weight[0]  ** 2

                    score = (weight_query_term * doc_weight[0])

                    if doc_id not in scores:
                        scores[doc_id] = score
                    else:
                        scores[doc_id] += score
                
            self.temp_index = {}
                
        # length normalize all scores
        for docID, score in scores.items():
            length = math.sqrt(docs_norm[docID]) * math.sqrt(query_len)
            #print(docID," -- ", score, length)
            scores[docID] /= length

        if self.boost_flag:
            for doc_id, arr in test_boost.items():
                if len(arr) > 1:
                    #Boost this docs
                    min_diff = self.getMinDiff(arr)
                    if min_diff:
                        boo = self.calc_boost(min_diff)
                        scores[doc_id] += boo

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return best_docs[:self.docs_limit] 
    

    def rank_bm25(self, indexed_query):
        scores = {}     #best docs
        test_boost = {}
        for term, tf in indexed_query.items():
            # Find where term is located
            right_index_file = self.findIndexFile(term)

            # idf for each term
            idf = self.dictionary[term][0]
            
            # weight = tf * idf 
            # logo, tf = weight / idf

            for doc_id, doc_weight in self.temp_index[right_index_file][term]['docs'].items():
                # Boost
                if doc_id not in test_boost:
                    test_boost[doc_id] = [doc_weight[1]]
                else:
                    test_boost[doc_id].append(doc_weight[1])

                doc_tf = doc_weight[0] / idf
                doc_length = self.doc_lengths[doc_id]
                score = self.calc_bm25(doc_tf, doc_length, self.avg_dl, idf)

                if doc_id not in scores:
                    scores[doc_id] = score
                else:
                    scores[doc_id] += score
        

        if self.boost_flag:
            for doc_id, arr in test_boost.items():
                if len(arr) > 1:
                    #Boost this docs
                    min_diff = self.getMinDiff(arr)
                    if min_diff:
                        boo = self.calc_boost(min_diff)
                        scores[doc_id] += boo

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return best_docs[:self.docs_limit]
    

    def calc_bm25(self, doc_tf, doc_length, avg_dl, idf):    
        return idf * (((self.k1 + 1) * doc_tf) / (self.k1 * ((1-self.b) + self.b * doc_length / avg_dl) + doc_tf))


    def calc_boost(self, min_range):
        return 12/math.log(1.5**min_range)


    def getMinDiff(self, lists):
        # invalid input
        if not lists:
            return None
    
        # `high` will be the maximum element in a heap
        high = -sys.maxsize
    
        # stores minimum and maximum elements found so far in a heap
        p = (0, sys.maxsize)
    
        # create an empty min-heap
        pq = []
    
        # push the first element of each lists into the min-heap
        # along with the lists number and their index in the lists
        for i in range(len(lists)):
            if not lists[i]:        # invalid input
                return None
            heappush(pq, Node(lists[i][0], i, 0))
            high = max(high, lists[i][0])
    
        # run till the end of any lists is reached
        while True:
    
            # remove the root node
            top = heappop(pq)
    
            # retrieve root node information from the min-heap
            low = top.value
            i = top.list_num
            j = top.index
    
            # update `low` and `high` if a new minimum is found
            if high - low < p[1] - p[0]:
                p = (low, high)
    
            # return on reaching the end of any lists
            if j == len(lists[i]) - 1:
                return p[1] - p[0]
    
            # take the next element from the "same" lists and
            # insert it into the min-heap
            heappush(pq, Node(lists[i][j + 1], i, j + 1))
    
            # update high if the new element is greater
            high = max(high, lists[i][j + 1])


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
                        if range not in self.temp_index.keys():
                            self.temp_index[range] = {}
                        self.temp_index[range][term_file] = { "docs": ast.literal_eval(value)}
                # we find the right index file
                break
        
        return right_index_file
    
    # Test
    def readFileMem(self, index_file):
        range = index_file.split('.')[0]
        with open("index/" + index_file) as f:
            for line in f.readlines():
                term_file,value = re.split('; ', line.rstrip('\n'), maxsplit=1)
                if range not in self.temp_index.keys():
                    self.temp_index[range] = {}
                self.temp_index[range][term_file] = { "docs": ast.literal_eval(value)}


    def evaluateQuery(self, query_relevance_docs, best_docs, time):
        docs_ids = best_docs[:50]

        tp = 0 # True Positives
        fp = 0 # False Positives
        fn = 0 # True Positives

        query_rel_docs = [doc[0] for doc in query_relevance_docs]
        query_docs = [doc[0] for doc in docs_ids]
        for doc in docs_ids:                
            if doc[0] in query_rel_docs:
                tp += 1

            if doc[0] not in query_rel_docs:
                fp += 1

        for doc in query_relevance_docs:                
            if doc[0] not in query_docs:
                fn += 1

        print("TP: {}   FP: {}  FN: {}".format(tp,fp,fn))

        precision = tp / (tp+fp)
        recall = tp / (tp+fn)
        if recall + precision == 0:
            f_score = 0
        else:
            f_score = (2 * recall * precision) / (recall + precision)

        print("Precision: {}   Recall: {}  F_score: {}".format(precision,recall,f_score))

    def readQueryRel(self):
        rel_queries = {}
        query_index = 0
        with open('queries/queries.relevance.txt','r') as q_rel:
            for line in q_rel.readlines():
                line = line.replace("\n", "")
                if 'Q:' in line:
                    query_index+=1
                #Read rel docs 
                elif line != "":
                    query_rel_list = line.split("\t")
                    if query_index in rel_queries.keys():
                        rel_queries[query_index].append((query_rel_list[0], query_rel_list[1]))
                    else:
                        rel_queries[query_index] = [(query_rel_list[0], query_rel_list[1])]

        return rel_queries


    # Version to read every index file that contains the term (correct version i guess)-- but the term was supposed to be in one file only...
    def rank_vector2(self, indexed_query):
        scores = {}     #best docs

        query_len = 0   # Norm query
        docs_norm = {}
        test_boost = {}
        for term, tf in indexed_query.items():
            # Find where term is located
            print("--> ",term)
            right_index_files = self.findIndexFile2(term)
            print(right_index_files)
            # idf for each term    
            idf = self.dictionary[term][0]
            
            tf_weight = math.log10(tf) + 1
            weight_query_term = tf_weight * float(idf) # Weight for the term in the query
            query_len += weight_query_term ** 2
            # Read every file that contains the term
            while right_index_files:
                right_index_file = right_index_files[0].split('.')[0]
                self.readFileMem(right_index_files[0])
                for doc_id, doc_weight in self.temp_index[right_index_file][term]['docs'].items():
                        # Boost
                        if self.boost_flag:
                            if doc_id not in test_boost:
                                test_boost[doc_id] = [doc_weight[1]]
                            else:
                                test_boost[doc_id].append(doc_weight[1])

                        # Norms
                        if doc_id not in docs_norm:
                            docs_norm[doc_id] = doc_weight[0] ** 2
                        else:
                            docs_norm[doc_id] += doc_weight[0]  ** 2

                        score = (weight_query_term * doc_weight[0])

                        if doc_id not in scores:
                            scores[doc_id] = score
                        else:
                            scores[doc_id] += score
                
                right_index_files.pop(0)

            self.temp_index = {}
                
        # length normalize all scores
        for docID, score in scores.items():
            length = math.sqrt(docs_norm[docID])
            #print(docID," -- ", score, length)
            scores[docID] /= length

        if self.boost_flag:
            for doc_id, arr in test_boost.items():
                if len(arr) > 1:
                    #Boost this docs
                    min_diff = self.getMinDiff(arr)
                    if min_diff:
                        boo = self.calc_boost(min_diff)
                        scores[doc_id] += boo

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return best_docs[:self.docs_limit] 


    def findIndexFile2(self, term):
        indexes_list = [file for file in os.listdir("index") if file != '.DS_Store']
        
        right_index_files = []
        for index_file in indexes_list:
            range = index_file.split('.')[0]
            first_word, last_word = range.split('_')[1:]
            if term >= first_word and term <= last_word:
                right_index_files.append(index_file)
                
        
        return right_index_files