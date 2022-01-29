import argparse
import ast
import sys
from tokenizer import Tokenizer
from indexer import Indexer
from ranker import Ranker
from sys import argv
import csv
import nltk
from nltk.corpus import stopwords
import time
import os

class SPIMI:
    def __init__(self, dataset, min_length, stopwords, limit=10000):
        self.dataset = dataset
        self.tokenizer = Tokenizer(min_length,stopwords)
        self.indexer = Indexer()
        self.chunk_limit = limit
        self.block_num = 1
        self.number_docs = 0


    def run(self):
        # Main function
        count = 0
        begin = time.time()
        
        with open(self.dataset,'r') as fd:
            rd = csv.DictReader(fd, delimiter="\t", quoting=csv.QUOTE_NONE)
            tokens = []
            print("Indexing...")
            for row in rd:
                self.number_docs += 1             
                # Shorter blocks mean a faster execution
                if count < self.chunk_limit:
                    review_id, product_title, review_headline, review_body = row['review_id'], row["product_title"], row['review_headline'], row['review_body']
                    string = product_title + " " + review_headline + " " + review_body
                    
                    tokens = self.tokenizer.get_tokens(string, review_id)
                    self.indexer.run(tokens)
                    count+=1
                     
                # reaching limit - write block on disk
                else:
                    # clear memory
                    tokens = []
                    # Check whether path exists or not
                    if not os.path.exists("blocks"):
                        # Create a new directory because it does not exist
                        os.makedirs("blocks")

                    self.indexer.write_block(self.block_num)
                    print((time.time()-begin)/60)
                    self.block_num += 1
                    
                    # clear memory
                    self.indexer.clear_index()
                    count=0

        if tokens != []:
            self.indexer.run(tokens) 
            if not os.path.exists("blocks"):
                # Create a new directory because it does not exist
                os.makedirs("blocks")
            self.indexer.write_block(self.block_num)
        
        
        # clear memory
        tokens = []
        self.indexer.clear_index()
        
        # For bm25
        self.tokenizer.writeDl()

        # Merge blocks  
        # Check whether path exists or not
        if not os.path.exists("index"):
            # Create a new directory because it does not exist
            os.makedirs("index")
        self.indexer.merge_blocks(self.number_docs)

        print("Total indexing time (min): ", round((time.time()-begin)/60, 2))
        print("Total index size on disk: ", self.indexer.get_index_size())
        print("Vocabulary size: ", self.indexer.get_vocabulary_size())
        print("Number of temporary index segments written to disk: ", self.block_num)



if __name__ == "__main__":
    default_stopwords = stopwords.words('english')
    # Command line arguments
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("dataset", help="Dataset")
    cli_parser.add_argument("-m", "--minimum", type=int, default=3, help="Minimum token length. Default 2 characets. Enter 0 to deactivate.")
    cli_parser.add_argument("-s", "--stopwords", default=None, help="Stopword list. Enter 'D' to deactivate")
    args = cli_parser.parse_args()
    
    data = args.dataset
    min_len = args.minimum
        
    if args.stopwords == None:
        stopwords = default_stopwords
    else:
        if args.stopwords == 'D':
            stopwords = args.stopwords
        else:
            stopwords = []
            with open(args.stopwords, 'r') as _file:
                for row in _file:
                    stopwords.append(row.strip())

    spimi = SPIMI(data, min_len,stopwords, 20000)
    spimi.run()