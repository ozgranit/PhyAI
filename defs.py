
#imports

import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy, os, sys, math, random, time, copy, argparse, platform, subprocess, logging, socket, time
from shutil import copyfile
import shutil
#from utils import create_job_file
from ete3 import *

DBSET = "1"
DIR_PREFIX = "" if DBSET == "1" else "DBset2/" if DBSET == "2" else "validation_set/" if DBSET== "val1" else "validation_set2/"
SEP = "/"

if platform.system() == 'Linux':
	DIRPATH = "/groups/itay_mayrose/danaazouri/PhyAI/"
	#DATA_PATH = SEP.join([DIRPATH+ "data", "training_datasets", ""])
else :
	if os.path.exists(r"D:\Users\Administrator\Dropbox"):	    # in lab
		DIRPATH = r"D:\Users\Administrator\Dropbox\PhyloAI\\"
	elif os.path.exists(r"C:\Users\ItayMNB3\Dropbox"):	    # laptop
		DIRPATH = r"C:\Users\ItayMNB3\Dropbox\PhyloAI\\"
	#DATA_PATH = SEP.join([DIRPATH, "data", ""])

DATA_PATH = SEP.join([DIRPATH+DIR_PREFIX + "data", "training_datasets", ""])
SUMMARY_FILES_DIR = SEP.join([DIRPATH+DIR_PREFIX, "summary_files", ""])
CODE_PATH = SEP.join([DIRPATH, "code", ""])

############# general
FASTA_FORMAT = "fasta"
PHYLIP_FORMAT = "phylip-relaxed"
NEXUS_FORMAT = "nexus"
MSA_PHYLIP_FILENAME_NOT_MASKED = "real_msa.phy"
MSA_PHYLIP_FILENAME = "masked_species_" + MSA_PHYLIP_FILENAME_NOT_MASKED
MSA_SUBTREE_FILENAME = "subtree_msa.phy"

REARRANGEMENTS_NAME = "rearrangement"
SUBTREE1 = "subtree1"
SUBTREE2 = "subtree2"
DBS = ['PANDIT', 'selectome', 'ploiDB', 'TreeBASE']
ROOTLIKE_NAME = "ROOT_LIKE"
SCORES_LST = ["corr", "best_predicted_ranking", "best_empirically_ranking", "required_evaluations_0.95"]
RANDOM_TREE_DIRNAME = "random_starting_tree/"
RANDOM_TREE_FILENAME = "starting_tree.txt"


############# summaries
CHOSEN_DATASETS_FILENAME = "sampled_datasets.csv"
SUMMARY_PER_DS = "{}ds_summary_{}_{}_step{}.csv"
TREES_PER_DS = "{}newicks_step{}.csv"
LEARNING_DATA_OLD = "learning_{}_step{}.csv"
LEARNING_DATA = "learning_{}_step{}.csv"
DATA_WITH_PREDS = "with_preds_merged_{}.csv"
SCORES_PER_DS = "scores_per_ds_{}.csv"


############# PHYML
PHYML_STATS_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_stats_{0}.txt"
PHYML_TREE_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_tree_{0}.txt"
MODEL_DEFAULT = "GTR+I+G"


############# learning
from collections import OrderedDict
FEATURES = OrderedDict([("bl", "edge_length"), ("longest", "longest_branch") ,
						("pdist_p", "pdist_average_pruned"), ("pdist_r", "pdist_average_remaining"),
						("ntaxa_p","ntaxa_prunned"),("ntaxa_r","ntaxa_remaining"),("tbl_p","tbl_pruned"),("tbl_r","tbl_remaining"),
						("pars_p","parsimony_pruned"),("pars_r","parsimony_remaining"),("longest_p","longest_pruned"),("longest_r","longest_remaining"),
						("top_dist","topology_dist_between"), ("bl_dist","tbl_dist_between"),   # only for rgft
						("res_tbl", "res_tree_tbl"), ("res_bl", "res_tree_edge_length"),        # only for rgft
						("group_id", "orig_ds_id"), ("group_tbl","orig_ds_tbl")])

FEATURES_RGFT_ONLY = ["top_dist", "bl_dist", "res_bl", "res_tbl"]
FEATURES_RGFT = [feature for key, feature in FEATURES.items()]
FEATURES_PRUNE = [feature for key, feature in FEATURES.items()]
[FEATURES_PRUNE.remove(FEATURES[f]) for f in FEATURES_RGFT_ONLY]

FEATURES_SHARED = ["orig_ds_id", "orig_ds_tbl", "longest_branch"]
merged_prune, merged_rgft = FEATURES_PRUNE.copy(), FEATURES_RGFT.copy()
[merged_prune.remove(f) for f in FEATURES_SHARED], [merged_rgft.remove(f) for f in FEATURES_SHARED]
FEATURES_MERGED = FEATURES_SHARED + [feature + "_prune" for feature in merged_prune] + [feature + "_rgft" for feature in merged_rgft]
LABEL = "d_ll_{}"

#NONIMPORTANT_FEATURES = ['ntaxa_prunned_rgft', 'ntaxa_prunned_prune', 'pdist_average_pruned_rgft',
#						 'pdist_average_pruned_prune', 'parsimony_pruned_rgft', 'parsimony_pruned_prune',
#						 'ntaxa_remaining_prune', 'ntaxa_remaining_rgft', 'pdist_average_remaining_rgft']




list_str = ['Source', 'path', 'Unnamed: 0.1_prune', 'prune_name', 'rgft_name', 'Unnamed: 0.1_rgft', 'Unnamed: 0', 'Unnamed: 0.1']
list_int = ['orig_ntaxa', 'ntaxa', 'nchars', 'orig_ds_id']
list_float = ['orig_ds_ll', 'll', 'd_ll_prune', 'edge_length_prune', 'longest_branch', 'ntaxa_prunned_prune', 'pdist_average_pruned_prune', 'tbl_pruned_prune', 'parsimony_pruned_prune', 'longest_pruned_prune', 'ntaxa_remaining_prune', 'pdist_average_remaining_prune', 'tbl_remaining_prune', 'parsimony_remaining_prune', 'longest_remaining_prune', 'orig_ds_tbl', 'd_ll_rgft', 'edge_length_rgft', 'ntaxa_prunned_rgft', 'pdist_average_pruned_rgft', 'tbl_pruned_rgft', 'parsimony_pruned_rgft', 'longest_pruned_rgft', 'ntaxa_remaining_rgft', 'pdist_average_remaining_rgft', 'tbl_remaining_rgft', 'parsimony_remaining_rgft', 'longest_remaining_rgft', 'topology_dist_between_rgft', 'tbl_dist_between_rgft', 'd_ll_merged', 'edge_length', 'ntaxa_prunned', 'pdist_average_pruned', 'tbl_pruned', 'parsimony_pruned', 'longest_pruned', 'ntaxa_remaining', 'pdist_average_remaining', 'tbl_remaining', 'parsimony_remaining', 'longest_remaining', 'topology_dist_between', 'tbl_dist_between', "res_tree_tbl_rgft", "res_tree_bl_rgft"]

types_dict = {}
for e in list_str:
	types_dict[e] = np.object
for e in list_int:
	types_dict[e] = np.int32
for e in list_float:
	types_dict[e] = np.float32





#################################3

def init_commandline_logger(logger):
	logger.setLevel(logging.DEBUG)
	# create console handler and set level to debug
	ch = logging.StreamHandler(sys.stdout)
	ch.setLevel(logging.DEBUG)
	# create formatter
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	# add formatter to ch
	ch.setFormatter(formatter)
	# add ch to logger
	logger.addHandler(ch)
