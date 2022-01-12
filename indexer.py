import ast
import os
import psutil
import math
from typing import cast
from more_itertools import locate



class Indexer:
    def __init__(self):
        self.indexed_tokens = {}
        self.vocabulary_size = set()
        self.index_size = 0
        self.index_num = 1
        self.number_docs = 0
        self.dictionary = {}

    def run(self, tokens):
        for token, _id in tokens:            
            if token not in self.indexed_tokens.keys():
                temp_dict = dict()            
                positions_list = list(locate(tokens, lambda x: x == (token, _id)))
                temp_dict[_id] = (tokens.count((token, _id)), positions_list)
                self.indexed_tokens[token] = temp_dict
            else:
                positions_list = list(locate(tokens, lambda x: x == (token, _id)))
                self.indexed_tokens[token][_id] = (tokens.count((token, _id)), positions_list)
    
    
    def clear_index(self):
        self.indexed_tokens = {}
    
    def get_indexed_tokens(self):
        return self.indexed_tokens
    
    def get_vocabulary_size(self):
        return len(self.vocabulary_size)

    def get_index_size(self):
        return self.index_size

    def write_block(self, number):
        print("Writing block...")
        sorted_index = dict(sorted(self.indexed_tokens.items()))
        with open("blocks/blocks_" + str(number) + ".txt",'w') as f:
            for token, value in sorted_index.items():
                string = token + ' : ' + str(value) + '\n'
                f.write(string)
        
        self.indexed_tokens = {}

    def merge_blocks(self, number_docs):
        self.number_docs = number_docs
        print("Merging...")
        temp_index = {}
        blocks_files = os.listdir("blocks")
        blocks_files = [open("blocks/"+block_file,'r') for block_file in blocks_files if block_file != '.DS_Store']
        lines = [(block_file.readline()[:-1], i) for i, block_file in enumerate(blocks_files)]
        initial_mem = psutil.virtual_memory().available

        while lines:
            for line, i in lines:

                line = line.split(" : ")
                if len(line) > 1:
                    term = line[0]
                    postings_dict = ast.literal_eval(line[1])

                    used_mem = initial_mem - psutil.virtual_memory().available                
                    if used_mem > 300000000:
                        print("Writing part of index...")
                        self.write_index(temp_index)
                        temp_index = {}
                        self.index_num+=1
                        initial_mem = psutil.virtual_memory().available

                    # Update index
                    if term in temp_index.keys():
                        for _id in postings_dict.keys():
                            if _id in temp_index[term]:
                                temp_index[term][_id] += postings_dict[_id]
                            else:
                                temp_index[term][_id] = postings_dict[_id]

                    else:
                        temp_index[term] = postings_dict
                    
                    # Update dictionary - idf
                    if term in self.dictionary.keys():
                        df = self.dictionary[term][1] + len(postings_dict.keys())
                        idf = math.log10(self.number_docs / df)
                        self.dictionary[term] = (idf, df)
                            
                    else:
                        df = len(postings_dict.keys())
                        idf = math.log10(self.number_docs / df)
                        self.dictionary[term] = (idf, df)
                            

            lines = [(block_file.readline()[:-1], i) for i, block_file in enumerate(blocks_files)]

            for line, i in lines:
                if not line:
                    blocks_files.pop(i)
                    lines.pop(i)

        print("Writing part of index...")
        self.write_dictionary()
        self.write_index(temp_index)
        temp_index = {}
        self.index_num+=1
    

    def write_dictionary(self):
        ordered_dict = dict(sorted(self.dictionary.items()))
        with open("dictionary/dictionary.txt",'w') as f:
            f.write(str(ordered_dict))


    def write_index(self, temp_index):
        ordered_dict = dict(sorted(temp_index.items()))
        first_word = list(ordered_dict.keys())[0]
        last_word = list(ordered_dict.keys())[-1]
        with open("index/index_"+str(first_word)+"_"+str(last_word)+".txt",'w') as f:
            for term, value in ordered_dict.items():
                new_post_list = {}
                # tf-idf weight
                for _id, freq in value.items():
                    tf = 1 + math.log10(freq)
                    idf = math.log10(self.number_docs / len(value))
                    weight = tf * idf
                    new_post_list[_id] = weight

                string = term + '; ' + str(new_post_list) + '\n'
                f.write(string)
        
        self.index_size += os.path.getsize('./index/index_' +str(first_word)+"_"+str(last_word)+ '.txt')
        self.vocabulary_size.update(list(ordered_dict.keys()))
        f.close()
        
    
    def term_query(self, query_term):
        index_files = os.listdir("index")
        index_files = [open("index/"+index_file,'r') for index_file in index_files if index_file != '.DS_Store']
        lines = [(index_file.readline()[:-1], i) for i, index_file in enumerate(index_files)]
        while lines:
            for line, i in lines:
                line = line.split("; ")
                term = line[0].split(": ")[0]
                postings_dict = ast.literal_eval(line[1])
                if len(line) > 1:
                    if query_term == term:
                        return len(postings_dict)

            lines = [(index_file.readline()[:-1], i) for i, index_file in enumerate(index_files)]
            for line, i in lines:
                if not line:
                    index_files.pop(i)
                    lines.pop(i)

        return "Term not found :("
        