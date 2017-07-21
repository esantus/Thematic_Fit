# LOADING LIBRARIES

import sys, codecs, os, gzip, re, pickle

from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting, PlmiWeighting
from composes.similarity.cos import CosSimilarity
from composes.matrix.sparse_matrix import SparseMatrix
from composes.utils import io_utils

from scipy import spatial
from scipy import stats
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools as itertools

from collections import defaultdict



# MODIFY THESE PATHS AND FILE NAMES ACCORDING TO YOUR WORKING ENVIRONMENT

matrix_path = "../matrices/"
dataset_path = "../datasets/"

filler_file = "../top_fillers/top_fillers.txt"
matrix_files = ["ukbnc1000_PPMI", "ukbnc1000_PLMI"]
dataset_files = ["mcrae.txt", "pado.txt", "instrum_ferretti.txt","locat_ferretti.txt", "obj_pado.txt", "sbj_pado.txt", "obj_mcrae.txt", "sbj_mcrae.txt"] 
output_file = "../results/output.txt"



# SET DEBUG TO TRUE IF YOU WANT TO SEE PRINTED ALL THE STEPS
DEBUG = False



# SYNTACTIC RELATIONS FILTER
syntactic_relations = {"all":["sbj-1", "obj-1", 'pmod+loc', 'pmod+with', 'pmod+on', 'pmod+in', 'pmod+at'], "pred":["sbj-1", "obj-1"], "compl":['pmod+with', 'pmod+on', 'pmod+in', 'pmod+at']}



# DICTIONARY OF PROTOTYPES
proto_mat = {}



# LOAD TOP FILLERS
top_fillers = load_top_fillers()



# OUTPUT FILE
output = open(output_file, "r")
    


# FOR EVERY MATRIX, FOR EVERY DATASET, FOR EVERY k...
for matrix in matrix_files:
    dsm = loadMat(matrix)

    for dataset_file in dataset_files:
        dataset = get_dataset(dataset_file)
        
        for number_of_fillers in range(10, 60, 20):


            # FOR EVERY TARGET VERB, CREATE A PROTOTYPE    
            for target in dataset.keys():
                temp = calculate_prototype(target, top_fillers, number_of_fillers, dsm)

                if temp == False:
                    if DEBUG == True:
                        print "Prototype for " + target + " was not created"
                    continue

                proto_mat[target] = temp

            # FOR EVERY TYPE OF CONTEXTS, CALCULATE SIMILARITY BETWEEN PROTOYPE AND CANDIDATE
            for type_of_contexts in syntactic_relations.keys():
                
                gold, cos, APSyn_500, APSyn_1000, APSyn_1500, APSyn_2000 = calculate_dataset(dataset, proto_mat, dsm, syntactic_relations[type_of_contexts])

                
                output.write(matrix + "\t" + dataset_file + "\t" + str(number_of_fillers) + "\t" + type_of_contexts + "\t" + str(stats.spearmanr(gold, cos)) + "\t" + str(stats.spearmanr(gold, APSyn_500)) + "\t" + str(stats.spearmanr(gold, APSyn_1000)) + "\t" + str(stats.spearmanr(gold, APSyn_1500)) + "\t" + str(stats.spearmanr(gold, APSyn_2000)))



# CLOSING THE OUTPUT FILE
output.close()
        
