import csv
from heapq import heappop, heappush
import imp
import math
import re
import os
import ast
import sys
import time
from node import Node
import numpy as np

class Ranker:
    def __init__(self, queries_file, tokenizer, boost_flag, norm_flag, method="vector", docs_limit=50, k1=1.2, b=0.75, doc_lengths=None, avg_dl=None):
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
        self.norm_flag = norm_flag

        # Evaluation means
        self.mean_precision_10, self.mean_recall_10, self.mean_f_measure_10, self.mean_avg_p_10, self.mean_ndcg_10 = [], [], [], [], []
        self.mean_precision_20, self.mean_recall_20, self.mean_f_measure_20, self.mean_avg_p_20, self.mean_ndcg_20 = [], [], [], [], []
        self.mean_precision_50, self.mean_recall_50, self.mean_f_measure_50, self.mean_avg_p_50, self.mean_ndcg_50 = [], [], [], [], []
        self.mean_latency = []
        self.mean_q_throuput = []

    def run(self):
        '''
        Process queries
        '''
        queries_list = self.readQueries()
        self.dictionary = self.readDictionary()

        # Read queries relevance to evaluate
        query_rel_list = self.readQueryRel()
        #res_query_list = self.readQueryRes()

        #Write header of evaluation
        self.writeEvalResults("header", [])
        for i in range(len(queries_list)):
            begin = time.time()
            indexed_query = self.query_freq(self.tokenizer.get_tokens(queries_list[i], i))

            print(indexed_query)
            print("Searching {}º query...".format(i+1))
            best_docs = []
            if self.method == 'vector':
                best_docs = self.rank_vector(indexed_query)
            elif self.method == 'bm25':
                best_docs = self.rank_bm25(indexed_query)

            # Evaluate query
            end = time.time()
            print("Time latency(sec): ", end-begin)
            print("Evaluating {}º query...".format(i+1))
            self.evaluateQuery(i+1,query_rel_list[i+1], best_docs, end - begin)
            self.queries_results[queries_list[i]] = best_docs
            
            #testing only first query 
            #break

        #Write evaluation means
        self.writeEvalResults("means", []) #Header = True

    def writeEvalResults(self, _type, content):
        if self.boost_flag:
            eval_file_path = "evaluation_results/eval_results_"+self.method+"_Boosted.csv"
        else:
            eval_file_path = "evaluation_results/eval_results_"+self.method+"_notBoosted.csv"

        if _type == "header":
            csv_file = open(eval_file_path, "w")
            writer = csv.writer(csv_file)
            writer.writerow(['', 'Precision', 'Precision', 'Precision', 'Recall', 'Recall', 'Recall', 'F-measure', 'F-measure', 'F-measure', 'AVG-Precision', 'AVG-Precision', 'AVG-Precision', 'NDCG', 'NDCG', 'NDCG', 'Latency','Query Throughput'])
            writer.writerow(['Query', '@10', '@20', '@50', '@10', '@20', '@50', '@10', '@20', '@50', '@10', '@20', '@50', '@10', '@20', '@50'])

            csv_file.close()
        elif _type == "content":
            csv_file = open(eval_file_path, "a")
            writer = csv.writer(csv_file)
            writer.writerow(content)

            csv_file.close()
        
        elif _type == "means":
            means = []
            means.append("Means")
            means.append(np.mean(self.mean_precision_10))
            means.append(np.mean(self.mean_precision_20))
            means.append(np.mean(self.mean_precision_50))

            means.append(np.mean(self.mean_recall_10))
            means.append(np.mean(self.mean_recall_20))
            means.append(np.mean(self.mean_recall_50))

            means.append(np.mean(self.mean_f_measure_10))
            means.append(np.mean(self.mean_f_measure_20))
            means.append(np.mean(self.mean_f_measure_50))

            means.append(np.mean(self.mean_avg_p_10))
            means.append(np.mean(self.mean_avg_p_20))
            means.append(np.mean(self.mean_avg_p_50))

            means.append(np.mean(self.mean_ndcg_10))
            means.append(np.mean(self.mean_ndcg_20))
            means.append(np.mean(self.mean_ndcg_50))

            means.append(np.mean(self.mean_latency))
            means.append(np.mean(self.mean_q_throuput))
            
            csv_file = open(eval_file_path, "a")
            writer = csv.writer(csv_file)
            writer.writerow(means)

            csv_file.close()

    def readDictionary(self):
        f = open("dictionary/dictionary.txt")
        return ast.literal_eval(f.read())

    def readQueries(self):
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
        
        if self.boost_flag:
            res_file_path = "search_output/results_"+ str(query_file.split('.')[0]) + "_" + str(self.method)+"_Boosted.txt"
        else:
            res_file_path = "search_output/results_"+ str(query_file.split('.')[0]) + "_" + str(self.method)+"_notBoosted.txt"
        with open(res_file_path,'w') as f:
            for query, doc_list in self.queries_results.items():
                f.write("Q:{}\n".format(query))
                for doc in doc_list:
                    f.write(str(doc)+"\t"+str(doc_list[doc])+"\n")
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
            print("-->: ",term)
            # Check if is in memory
            if term not in self.temp_index:
                print("Term not in memory")
                self.temp_index={}
                # Find where term is located
                self.findIndexFile(term)
            # idf for each term    
            idf = self.dictionary[term][0]
            
            tf_weight = math.log10(tf) + 1
            weight_query_term = tf_weight * float(idf) # Weight for the term in the query
            query_len += weight_query_term ** 2

            for doc_id, doc_weight in self.temp_index[term]['docs'].items():
                    # Boost
                    if self.boost_flag:
                        if doc_id not in test_boost:
                            test_boost[doc_id] = [doc_weight[1]]
                        else:
                            test_boost[doc_id].append(doc_weight[1])

                    # Norms
                    if self.norm_flag:
                        if doc_id not in docs_norm:
                            docs_norm[doc_id] = doc_weight[0] ** 2
                        else:
                            docs_norm[doc_id] += doc_weight[0]  ** 2

                    score = (weight_query_term * doc_weight[0])

                    if doc_id not in scores:
                        scores[doc_id] = score
                    else:
                        scores[doc_id] += score
                
                
        # length normalize all scores
        if self.norm_flag:
            for docID, score in scores.items():
                length = math.sqrt(docs_norm[docID]) * math.sqrt(query_len)
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
        return dict(best_docs[:self.docs_limit])
    

    def rank_bm25(self, indexed_query):
        scores = {}     #best docs
        test_boost = {}
        for term, tf in indexed_query.items():
            print("-->: ",term)
            # Check if is in memory
            if term not in self.temp_index:
                print("Term not in memory")
                self.temp_index={}
                # Find where term is located
                self.findIndexFile(term)
            # idf for each term
            idf = self.dictionary[term][0]
            
            # weight = tf * idf 
            # logo, tf = weight / idf

            for doc_id, doc_weight in self.temp_index[term]['docs'].items():
                # Boost
                if self.boost_flag:
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
        return dict(best_docs[:self.docs_limit])
    

    def calc_bm25(self, doc_tf, doc_length, avg_dl, idf):    
        return idf * (((self.k1 + 1) * doc_tf) / (self.k1 * ((1-self.b) + self.b * doc_length / avg_dl) + doc_tf))


    def calc_boost(self, min_range):
        return 10/math.log(1.5**min_range)


    # Get smallest diference - https://www.techiedelight.com/find-smallest-range-least-one-element-given-lists/
    def getMinDiff(self, lists):    
        '''
        Get smallest diference - https://www.techiedelight.com/find-smallest-range-least-one-element-given-lists/
        '''
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
        '''
        Get the index file that contains the term and bring it to memory
        '''
        indexes_list = [file for file in os.listdir("index") if file != '.DS_Store']
        
        right_index_file = None
        for index_file in indexes_list:
            range = index_file.split('.')[0]
            first_word, last_word = range.split('_')[1:]
            if term >= first_word and term <= last_word:
                right_index_file = range
                print(right_index_file)
                with open("index/" + index_file) as f:
                    for line in f.readlines():
                        print(line)
                        term_file,value = re.split('; ', line.rstrip('\n'), maxsplit=1)
                        self.temp_index[term_file] = { "docs": ast.literal_eval(value)}
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


    def evaluateQuery(self, index, query_relevance_docs, best_docs, time):
        # query_relevance_docs = { query1_id : { doc_1: relevance, doc_2: relevance,...},...}
        # best_docs = { query1_id : { doc_1: score, doc_2: score,...},...}
        docs_eval_num = [10,20,50]

        for num in docs_eval_num:
            tp = 0 # True Positives
            fp = 0 # False Positives
            fn = 0 # True Positives

            precisions_avg = []
            cg_list = []
            docs_ids = dict(list(best_docs.items())[:num])
            print("\nNUM--------------: ",num)
            for doc in docs_ids:    
                # True Pos            
                if doc in query_relevance_docs:
                    tp += 1
                    #AVG Precision
                    temp_avg_p = tp / (tp+fp)
                    precisions_avg.append(temp_avg_p)

                    rel = query_relevance_docs[doc]
                    cg_list.append(int(rel))  

                # False Pos  
                else:
                    fp += 1

            for doc in query_relevance_docs:     
                # False Neg            
                if doc not in docs_ids:
                    fn += 1

            print("TP: {}   FP: {}  FN: {}".format(tp,fp,fn))

            precision = tp / (tp+fp)
            recall = tp / (tp+fn)
            if precision + recall == 0:
                f_measure = 0
            else:
                f_measure = (2 * precision * recall) / (precision + recall)

            if tp != 0:
                avg_p = sum(precisions_avg) / tp
            else:
                avg_p = 0

            #NDCG
            dcg_list = [rel/math.log2((i+1) + 1) for i, rel in enumerate(cg_list)]
            idcg_list = [rel/math.log2((i+1) + 1) for i, rel in enumerate(sorted(cg_list, reverse=True))]
            dcg = sum(dcg_list)
            idcg = sum(idcg_list)
            # Normalized DCG score
            if idcg == 0:
                ndcg = 0
            else:
                ndcg = dcg / idcg

            if num==10:
                    recall_10 = recall
                    precision_10 = precision
                    f_10 = f_measure
                    ap_10 = avg_p
                    ndcg_10 = ndcg

                    # add to mean lists
                    self.mean_precision_10.append(precision)
                    self.mean_recall_10.append(recall)
                    self.mean_f_measure_10.append(f_measure)
                    self.mean_avg_p_10.append(avg_p)
                    self.mean_ndcg_10.append(ndcg)

            elif num==20:
                    recall_20 = recall
                    precision_20 = precision
                    f_20 = f_measure
                    ap_20 = avg_p
                    ndcg_20 = ndcg

                    # add to mean lists
                    self.mean_precision_20.append(precision)
                    self.mean_recall_20.append(recall)
                    self.mean_f_measure_20.append(f_measure)
                    self.mean_avg_p_20.append(avg_p)
                    self.mean_ndcg_20.append(ndcg)
            elif num==50:
                    recall_50 = recall
                    precision_50 = precision
                    f_50 = f_measure
                    ap_50 = avg_p
                    ndcg_50 = ndcg

                    # add to mean lists
                    self.mean_precision_50.append(precision)
                    self.mean_recall_50.append(recall)
                    self.mean_f_measure_50.append(f_measure)
                    self.mean_avg_p_50.append(avg_p)
                    self.mean_ndcg_50.append(ndcg)
        
        self.mean_latency.append(time)
        self.mean_q_throuput.append(1/time)
        content_list = [index, precision_10,precision_20,precision_50, recall_10, recall_20, recall_50, f_10, f_20, f_50 \
                ,ap_10,ap_20,ap_50, ndcg_10, ndcg_20, ndcg_50, time, 1/time]
        self.writeEvalResults("content",content_list)
        

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
                        rel_queries[query_index][query_rel_list[0]] = query_rel_list[1]
                    else:
                        rel_queries[query_index] = {}
                        rel_queries[query_index][query_rel_list[0]] = query_rel_list[1]

        return rel_queries


    #Test for evaluation
    def readQueryRes(self):
        rel_queries = {}
        query_index = 0
        with open('search_output/results_queries_bm25.txt','r') as q_rel:
            for line in q_rel.readlines():
                line = line.replace("\n", "")
                if 'Q:' in line:
                    query_index+=1
                #Read rel docs 
                elif line != "":
                    query_rel_list = line.split("\t")
                    if query_index in rel_queries.keys():
                        rel_queries[query_index][query_rel_list[0]] = query_rel_list[1]
                    else:
                        rel_queries[query_index] = {}
                        rel_queries[query_index][query_rel_list[0]] = query_rel_list[1]

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
                        if self.norm_flag:
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
        if self.norm_flag:
            for docID, score in scores.items():
                length = math.sqrt(docs_norm[docID])
                #print(docID," -- ", score, length)
                scores[docID] /= length

        if self.boost_flag:
            for doc_id, arr in test_boost.items():
                if len(arr) > 1:
                    #Boost this docs
                    print("Boost: ",doc_id)
                    min_diff = self.getMinDiff(arr)
                    if min_diff:
                        boo = self.calc_boost(min_diff)
                        scores[doc_id] += boo

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(best_docs[:self.docs_limit])


    def findIndexFile2(self, term):
        indexes_list = [file for file in os.listdir("index") if file != '.DS_Store']
        
        right_index_files = []
        for index_file in indexes_list:
            range = index_file.split('.')[0]
            first_word, last_word = range.split('_')[1:]
            if term >= first_word and term <= last_word:
                right_index_files.append(index_file)
        
        return right_index_files


    def rank_bm25_2(self, indexed_query):
        scores = {}     #best docs
        test_boost = {}
        for term, tf in indexed_query.items():
            # Find where term is located
            right_index_files = self.findIndexFile2(term)
            print(right_index_files)
            # idf for each term
            idf = self.dictionary[term][0]
            
            # weight = tf * idf 
            # logo, tf = weight / idf
            # Read every file that contains the term
            while right_index_files:
                right_index_file = right_index_files[0].split('.')[0]
                self.readFileMem(right_index_files[0])
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
        
                right_index_files.pop(0)
            
                self.temp_index={}

        if self.boost_flag:
            for doc_id, arr in test_boost.items():
                if len(arr) > 1:
                    #Boost this docs
                    min_diff = self.getMinDiff(arr)
                    if min_diff:
                        boo = self.calc_boost(min_diff)
                        scores[doc_id] += boo

        best_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return dict(best_docs[:self.docs_limit])