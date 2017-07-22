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



def get_vec(target, dsm):
    
    if is_context(target) == True:
        target = context2target(target)
        
    target2index = {w: i for i, w in enumerate(dsm.id2row)}
    index = target2index.get(target.lower(), -1)
    
    if index == -1:
        if DEBUG == True:
            print "WARNING: ", target, " does not exist in the matrix! (GET_VEC)"
        return False
    
    cooc_mat = dsm.cooccurrence_matrix
    return cooc_mat[index, :]



def is_context(target):
    
    if DEBUG == True:
        print "We are checking whether ", target, " can be splint in more than one piece by \':\'!"
    
    if len(target.split(":")) > 1:
        return True
    return False



def context2target(context):
    
    if DEBUG == True:
        print "We are returning the following target from a context: ", target.split(":")[1]
    
    return target.split(":")[1]



def dataset_target2target_role(target):
    
    target = target.split("_")
    
    if DEBUG == True:
        print "We are returning the following target and role from a dataset_target: ", target[0], ", ", target[1]
    
    return target[0], target[1]



def load_top_fillers():
    
    top_fillers = {}
    sbj_dic = {}
    loc_dic = {}
    
    with open(home+filler_path+filler_file, "r") as top_fillers_f:
        
        for line in top_fillers_f:
            line = line.strip().split("\t")
            
            if line[0] not in top_fillers.keys():
                top_fillers[line[0]] = []
                
            found_sbj = False

            if line[1] == "sbj_tr-1" or line[1]== "sbj_intr-1":
                if (line[0]+"_"+line[2]+"sbj") not in sbj_dic:
                    sbj_dic[line[0]+"_"+line[2]+"sbj"] = 1
                else:
                    found_sbj = True
                                        
            found_loc = False

            if line[1] == "at-1" or line[1]== "on-1" or line[1]== "in-1":
                if (line[0]+"_"+line[2]+"loc") not in loc_dic:
                    loc_dic[line[0]+"_"+line[2]+"loc"] = 1
                else:
                    found_loc = True
                    
            if found_sbj == False and found_loc == False:
                top_fillers[line[0]].append((line[1], line[2], line[3]))
    
    if DEBUG == True:
        for key in top_fillers:
            print key, " = ", top_fillers[key], "\n\n\n"
            
    return top_fillers



def filter_top_fillers(top_fillers, role_filter, N):
    
    for key in top_fillers:
        top_fillers[key] = [(role, target, score)  for role, target, score in top_fillers[key] if role in role_filter]
        top_fillers[key] = top_fillers[key][:(min(len(top_fillers[key]), N))]
    
    if DEBUG == True:
        for key in top_fillers:
            print key, " = ", top_fillers[key], "\n\n\n"
            
    return top_fillers  



def calculate_prototype(target, top_fillers, N, dsm):
    
    proto = []
    
    target, target_role = dataset_target2target_role(target)

    role = {"sbj":["sbj_tr-1", "sbj_intr-1"], "obj":["obj-1"], "pmod+loc":["in-1", "on-1", "at-1"], "pmod+with":["with-1"]}
    
    if DEBUG == True:
        print target, role

    if target in top_fillers.keys():
        
        vecs = [v_target for v_role, v_target, v_score in top_fillers[target] if v_role in role[target_role]]

        if DEBUG == True:
            print vecs

        if len(vecs) < N:
            print "WARNING: ", target, " does not have enough fillers for the role ", role[target_role]
            return False

        if DEBUG == True:
            print "THIS IS NOT A WARNING: ", target, " has enough fillers for the role ", role[target_role]
            
        i = 0
        
        while i < N:
            
            if DEBUG == True:
                print "Getting vector of filler ", vecs[i]
                
            temp = get_vec(vecs[i], dsm)
            
            if temp == False:
                i += 1
                N += 1
                continue
            
            if proto == []:
                proto = temp
                #print "Proto: ", type(proto), len(proto)
            else:
                proto = proto + temp
                #print "Proto: ", type(proto), len(proto)
                
            i += 1

        return proto

    else:
        print "WARNING: ", target, " does not exist in the filler database."
        return False



def loadMat(dsm_prefix):
    dsm_prefix = project + matrix_path + dsm_prefix

    print("Loading the Matrix")
    return load_pkl_files(dsm_prefix)


def load_pkl_files(dsm_prefix):
    """
    Load the space from either a single pkl file or numerous files
    :param dsm_prefix:
    :param dsm:
    :return:
    """
    # Check whether there is a single pickle file for the Space object
    if os.path.isfile(dsm_prefix + '.pkl'):
        return io_utils.load(dsm_prefix + '.pkl')

    # Load the multiple files: npz for the matrix and pkl for the other data members of Space
    with np.load(dsm_prefix + 'cooc.npz') as loader:
        coo = coo_matrix((loader['data'], (loader['row'], loader['col'])), shape=loader['shape'])

    cooccurrence_matrix = SparseMatrix(csr_matrix(coo))

    with open(dsm_prefix + '_row2id.pkl', 'rb') as f_in:
        row2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2row.pkl', 'rb') as f_in:
        id2row = pickle.load(f_in)

    with open(dsm_prefix + '_column2id.pkl', 'rb') as f_in:
        column2id = pickle.load(f_in)

    with open(dsm_prefix + '_id2column.pkl', 'rb') as f_in:
        id2column = pickle.load(f_in)

    return Space(cooccurrence_matrix, id2row, id2column, row2id=row2id, column2id=column2id)



def get_dataset(file_name):
    
    dataset = {}
    
    with open(home+dataset_path+file_name, "r") as dataset_f:
        for line in dataset_f:
            item = line.strip().split("\t")
            
            if item[0] not in dataset.keys():
                dataset[item[0]] = []

            dataset[item[0]].append((item[1], item[2]))
            
        return dataset



def cosine(v1, v2):
    if v1.norm() == 0 or v2.norm() == 0:
        return 0.0

    return v1.multiply(v2).sum() / np.double(v1.norm() * v2.norm())



def calculate_dataset(dataset, proto_mat, dsm, roles_to_be_considered):
    
    gold = []
    cos = []
    APSyn_500 = []
    APSyn_1000 = []
    APSyn_1500 = []
    APSyn_2000 = []
    
    for proto_target in dataset.keys():
        
        if proto_target in proto_mat.keys():
            for filler in dataset[proto_target]:
                
                if DEBUG == True:
                    print "The filler is: ", filler
                
                filler_vec = get_vec(filler[0], dsm)

                if filler_vec == False:
                    if DEBUG == True:
                        print filler[0], ", from the dataset, does not exist in the matrix!"
                    continue
                
                if DEBUG == True:
                    print "proto_mat[proto_target]", type(proto_mat[proto_target])
                    print "filler_vec", type(filler_vec)
                
                vector_cosine = cosine(proto_mat[proto_target], filler_vec)
                vector_APSyn = APSyn(proto_mat[proto_target], filler_vec, 500, dsm, roles_to_be_considered)
                APSyn_500.append(vector_APSyn)
                vector_APSyn = APSyn(proto_mat[proto_target], filler_vec, 1000, dsm, roles_to_be_considered)
                APSyn_1000.append(vector_APSyn)
                vector_APSyn = APSyn(proto_mat[proto_target], filler_vec, 1500, dsm, roles_to_be_considered)
                APSyn_1500.append(vector_APSyn)
                vector_APSyn = APSyn(proto_mat[proto_target], filler_vec, 2000, dsm, roles_to_be_considered)
                APSyn_2000.append(vector_APSyn)
                
                gold.append(float(filler[1]))
                cos.append(vector_cosine)
                
                if DEBUG == True:
                    print proto_target + "\t" + filler[0] + "\t" + filler[1] + "\t" + str(vector_cosine) + "\t" + str(vector_APSyn)
                
        else:
            print "WARNING: ", proto_target, " does not exist in the proto_matrix."
    
    print gold, cos, APSyn_500, APSyn_1000, APSyn_1500, APSyn_2000
    return gold, cos, APSyn_500, APSyn_1000, APSyn_1500, APSyn_2000



def sort_by_value_get_col(mat):
    
    tuples = izip(mat.row, mat.col, mat.data)
    sorted_tuples = sorted(tuples, key=lambda x: x[2], reverse=True)

    if len(sorted_tuples) == 0:
        return []

    rows, columns, values = zip(*sorted_tuples)
    return columns


def APSyn(x_row, y_row, N, dsm, roles):
    
    index2context = {i: w for i, w in enumerate(dsm.id2column)}

    new_y_contexts_cols = []
    new_x_contexts_cols = []

    print_contexts = False
    
    # Sort y's contexts
    y_contexts_cols = sort_by_value_get_col(scipy.sparse.coo_matrix(y_row.mat)) # tuples of (row, col, value)

    if print_contexts == True:
        for o in range(0, 10):
            if len(y_contexts_cols) > o:
                print "\t", index2context.get(y_contexts_cols[o][1], "NOTHING")

        print "\n"

    m = 0
    for tuple in y_contexts_cols:

        context = index2context.get(tuple, "FALSE")

        if context != "FALSE":
            try:
                role, context = context.split(":")
                role = role.lower()
            except:
                print("This has more :, ahi: %s " % context)
                continue

            if role in roles: 
                new_y_contexts_cols.append(tuple)
                m += 1
                
                if m >= N:
                    break
        else:
            print("Context does not exist in the matrix")

    y_context_rank = { c : i + 1 for i, c in enumerate(new_y_contexts_cols) }

    # Sort x's contexts
    x_contexts_cols = sort_by_value_get_col(scipy.sparse.coo_matrix(x_row.mat))

    if print_contexts == True:
        for o in range(0, 10):
            if len(x_contexts_cols) > o:
                print "\t", index2context.get(x_contexts_cols[o][1], "NOTHING")

    m = 0
    for tuple in x_contexts_cols:

        context = index2context.get(tuple, "FALSE")

        if context != "FALSE":
            try:
                role, context = context.split(":")
            except:
                print("This has more :, ahi: %s " % context)
                continue
    
            if role in roles:
                new_x_contexts_cols.append(tuple)
                m += 1
                

                if m >= N:
                    break
        else:
            print("Context does not exist in the matrix")
            
    x_context_rank = { c : i + 1 for i, c in enumerate(new_x_contexts_cols) }

    intersected_context = set(new_y_contexts_cols).intersection(set(new_x_contexts_cols))

    score = sum([1.0 / ((x_context_rank[c] + y_context_rank[c]) / 2.0) for c in intersected_context])

    if print_contexts == True:
        print "APSyn: " + str(score) + "\n"
        for o in range(0, 10):
            if len(intersected_context) > o:
                print "Intersected: ", list(intersected_context)[o], index2context.get(list(intersected_context)[o], "NOTHING")
        print "\n\n"

    return score




















