import argparse
import ast
import sys
import time
from nltk.corpus import stopwords

from ranker import Ranker
from tokenizer import Tokenizer


def search(tokenizer, ranking, queries, norm_flag):     
    doc_lengths = None
    avg_dl = None
    begin = time.time()
    if flag:
        try:
            f = open("dl.txt")
            doc_lengths = ast.literal_eval(f.read())
            avg_dl = sum(list(doc_lengths.values())) / len(doc_lengths.keys())
        except:
            print("Run indexer")
            sys.exit()

    ranker = Ranker(queries, tokenizer, boost_flag, norm_flag, ranking, 50, 1.2, 0.75, doc_lengths, avg_dl)
    
    ranker.run()
    print("Writing results...")
    ranker.writeResults()
    print("Total searching and writing time (min): ", round((time.time()-begin)/60, 2)) 

if __name__ == "__main__":
    default_stopwords = stopwords.words('english')
    # Command line arguments
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument("-m", "--minimum", type=int, default=3, help="Minimum token length. Default 2 characets. Enter 0 to deactivate.")
    cli_parser.add_argument("-s", "--stopwords", default=None, help="Stopword list. Enter 'D' to deactivate")
    cli_parser.add_argument("-r", "--ranking", type=str, default='vector',
                            help="Ranking algorithm. \n->\"vector\" for TF-IDF \n->\"bm25\" for BM25. Default is TD-IDF.")
    cli_parser.add_argument("-q", "--queries", default='queries/queries.txt', help="Select the file from which the queries are read. Default file is \'queries.txt\'")
    cli_parser.add_argument("-b", "--boost", default='false', help="With or without boost - true or false. Defualt false.")
    cli_parser.add_argument("-n", "--norm", default='true', help="With or without score normalization - true or false. Defualt true.")
    args = cli_parser.parse_args()
    
    min_len = args.minimum
    boost_flag = False
    if args.boost.lower() == "true":
        print("Boost activated!")
        boost_flag = True

    norm_flag = False
    if args.norm.lower() == "true":
        norm_flag = True

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

    ranking = args.ranking
    flag = False
    if ranking == 'bm25':
        flag = True

    query_file = args.queries
    tokenizer = Tokenizer(min_len,stopwords)
    if ranking == "vector" or ranking == "bm25":
        search(tokenizer, ranking, query_file, norm_flag)
    else:
        print("Ranking method not available!")