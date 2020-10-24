import os
import re
import shutil
import sys

import networkx
import numpy
import pandas as pd
import pylab

from Bio import Phylo
from ete3 import Tree, PhyloTree
from subprocess import Popen, PIPE, STDOUT

from pathlib import Path

parent_path = Path().resolve().parent

parent_folder = parent_path / "reinforcement_data"


RAXML_NG_SCRIPT = "raxml-ng"    # after you install raxml-ng on your machine
# conda install -c bioconda raxml-ng
MSA_PHYLIP_FILENAME = "masked_species_real_msa.phy"


def return_likelihood(tree_str, msa_file, rates, pinv, alpha, freq):
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
									 pinv="{{{0}}}".format(pinv), alpha="{{{0}}}".format(alpha),
									 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/dev/shm/" + msa_file.split("/")[-1] + "tree"  # the var is the str: tmp{dir_suffix}
	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree_str)

		p = Popen([RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_file,'--threads', '1', '--opt-branches', 'on', '--opt-model', 'off', '--model', model_line_params, '--nofiles', '--tree', tree_rampath],
				  stdout=PIPE, stdin=PIPE, stderr=STDOUT)
		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()

		res_dict = parse_raxmlNG_content(raxml_output)
		ll = res_dict['ll']
		rtime = res_dict['time']

	except Exception as e:
		print(msa_file.split("/")[-1])
		print(e)
		exit()
	finally:
		os.remove(tree_rampath)

	return ll, rtime


def parse_raxmlNG_output(res_filepath):

	try:
		with open(res_filepath) as fpr:
			content = fpr.read()
		res_dict = parse_raxmlNG_content(content)
	except:
		print("Error with:", res_filepath)
		return

	return res_dict


def parse_raxmlNG_content(content):
	"""
	:return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
	"""
	res_dict = dict.fromkeys(["ll", "pInv", "gamma",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "time"], "")

	# likelihood
	ll_re = re.search("Final LogLikelihood:\s+(.*)", content)
	if not ll_re and (re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content)):
		res_dict["ll"] = re.search("initial LogLikelihood:\s+(.*)", content).group(1).strip()
	else:
		res_dict["ll"] = ll_re.group(1).strip()

		# gamma (alpha parameter) and proportion of invariant sites
		gamma_regex = re.search("alpha:\s+(\d+\.?\d*)\s+", content)
		pinv_regex = re.search("P-inv.*:\s+(\d+\.?\d*)", content)
		if gamma_regex:
			res_dict['gamma'] = gamma_regex.group(1).strip()
		if pinv_regex:
			res_dict['pInv'] = pinv_regex.group(1).strip()

		# Nucleotides frequencies
		nucs_freq = re.search("Base frequencies.*?:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
		for i,nuc in enumerate("ACGT"):
			res_dict["f" + nuc] = nucs_freq.group(i+1).strip()

		# substitution frequencies
		subs_freq = re.search("Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)", content)
		for i,nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):  # todo: make sure order
			res_dict["sub" + nuc_pair] = subs_freq.group(i+1).strip()

		# Elapsed time of raxml-ng optimization
		rtime = re.search("Elapsed time:\s+(\d+\.?\d*)\s+seconds", content)
		if rtime:
			res_dict["time"] = rtime.group(1).strip()
		else:
			res_dict["time"] = 'no ll opt_no time'
	return res_dict


def parse_phyml_stats_output(stats_filepath):
	"""
    :return: dictionary with the attributes - string typed. if parameter was not estimated, empty string
    """
	res_dict = dict.fromkeys(["ntaxa", "nchars", "ll",
							  "fA", "fC", "fG", "fT",
							  "subAC", "subAG", "subAT", "subCG", "subCT", "subGT",
							  "pInv", "gamma",
							  "path"], "")


	res_dict["path"] = stats_filepath
	try:
		with open(stats_filepath) as fpr:
			content = fpr.read()

		# likelihood
		res_dict["ll"] = re.search("Log-likelihood:\s+(.*)", content).group(1).strip()

		# gamma (alpha parameter) and proportion of invariant sites
		gamma_regex = re.search("Gamma shape parameter:\s+(.*)", content)
		pinv_regex = re.search("Proportion of invariant:\s+(.*)", content)
		if gamma_regex:
			res_dict['gamma'] = gamma_regex.group(1).strip()
		if pinv_regex:
			res_dict['pInv'] = pinv_regex.group(1).strip()

		# Nucleotides frequencies
		for nuc in "ACGT":
			nuc_freq = re.search("  - f\(" + nuc + "\)\= (.*)", content).group(1).strip()
			res_dict["f" + nuc] = nuc_freq

		# substitution frequencies
		for nuc1 in "ACGT":
			for nuc2 in "ACGT":
				if nuc1 < nuc2:
					nuc_freq = re.search(nuc1 + " <-> " + nuc2 + "(.*)", content).group(1).strip()
					res_dict["sub" + nuc1 + nuc2] = nuc_freq
	except:
		print("Error with:", res_dict["path"], res_dict["ntaxa"], res_dict["nchars"])
		return
	return res_dict


def prune_branch(t_orig, prune_name):
	'''
	get (a copy of) both subtrees after pruning
	'''
	t_cp_p = t_orig.copy()  # the original tree is needed for each iteration
	assert t_cp_p & prune_name    # todo Oz: add indicative error
	prune_node_cp = t_cp_p & prune_name  # locate the node in the copied subtree
	assert prune_node_cp.up

	nname = prune_node_cp.up.name
	prune_loc = prune_node_cp
	prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
	t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

	return nname, prune_node_cp, t_cp_p


def regraft_branch(t_cp_p, prune_node_cp, rgft_name, nname):
	'''
	get a tree with the 2 concatenated subtrees
	'''

	t_temp = PhyloTree()  # for concatenation of both subtrees ahead, to avoid polytomy
	t_temp.add_child(prune_node_cp)
	t_curr = t_cp_p.copy()
	assert t_curr & rgft_name   # todo Oz: add indicative error
	rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree
	new_branch_length = rgft_node_cp.dist / 2

	rgft_loc = rgft_node_cp.up
	rgft_node_cp.detach()
	t_temp.add_child(rgft_node_cp, dist=new_branch_length)
	t_temp.name = nname
	rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting

	return t_curr


def SPR_by_edge_names(tree_obj, cut_name, paste_name):
	nname, subtree1, subtree2 = prune_branch(tree_obj, cut_name)  # subtree1 is the pruned subtree. subtree2 is the remaining subtree
	rearr_tree_str = regraft_branch(subtree2, subtree1, paste_name, nname).write(format=1)

	return rearr_tree_str


def add_internal_names(tree_file, tree_file_cp_no_internal, t_orig):
	shutil.copy(tree_file, tree_file_cp_no_internal)
	for i, leaf in enumerate(t_orig.traverse()):
			leaf.name = "N{}".format(i)
	t_orig.write(format=3, outfile=tree_file)   # runover the orig file with no internal nodes names


# convert tree to weighted_adjacency_matrix
def tree_to_matrix(tree):
	net = Phylo.to_networkx(tree)
	pos = networkx.spring_layout(net)
	# print(pos)
	# networkx.draw(net)
	matrix = networkx.adjacency_matrix(net)
	return matrix.toarray()
	# pylab.show()


# returns the tree from the text file in the msa_num's folder
def get_tree_from_msa(msa_path="/data/training_datasets/82/"):
	tree_path = parent_folder / (msa_path + "masked_species_real_msa.phy_phyml_tree_bionj.txt")
	tree = Phylo.read(tree_path, "newick")
	return tree


# calculating likelihood of tree, msa_num should be the folder number of its corresponding msa
def get_likelihood_simple(tree, msa_path="/data/training_datasets/82/"):

	# taking the params required for likelihood calculation from the stats file in the msa_num's folder
	stats_path = parent_folder / (msa_path + "masked_species_real_msa.phy_phyml_stats_bionj.txt")
	params_dict = parse_phyml_stats_output(stats_path)
	freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"], params_dict["subCT"],params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

	msa_path = parent_folder / (msa_path + "masked_species_real_msa.phy")
	return return_likelihood(tree, msa_path, rates, pinv, alpha, freq)


if __name__ == '__main__':
	# update to full path
	df = pd.read_csv("sampled_datasets.csv")
	# test on 1 dataset only
	curr_path = df.loc[0, "path"]
	tree_path = curr_path + MSA_PHYLIP_FILENAME + "_phyml_tree_bionj.txt"
	t_orig = Tree(newick=tree_path, format=1)
	t_orig.get_tree_root().name = "ROOT_LIKE"

	tree_file_cp_no_internal =  tree_path + "_no_internat.txt"
	add_internal_names(tree_path, tree_file_cp_no_internal, t_orig)

	'''
	lst_of_possible_cut_and_paste_names_pairs = []    # you could generate as following:
	for i, prune_node in enumerate(t_orig.iter_descendants("levelorder")):
		prune_name = prune_node.name
		nname, subtree1, subtree2 = prune_branch(t_orig, prune_name)
		for j, rgft_node in enumerate(subtree2.iter_descendants("levelorder")):
			rgft_name = rgft_node.name
			#lst_of_possible_cut_and_paste_names_pairs.append(...
	neighbor_tree_str = SPR_by_edge_names(t_orig, lst_of_possible_cut_and_paste_names_pairs[0][0], lst_of_possible_cut_and_paste_names_pairs[0][1])
	'''

	neighbor_tree_str = SPR_by_edge_names(t_orig, 'Sp000', 'Sp001')

	# extract model params from the starting tree, to fix when calculating the likelihood of all neighbors
	params_dict = parse_phyml_stats_output(curr_path + MSA_PHYLIP_FILENAME + "_phyml_stats_bionj.txt")
	freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"], params_dict["subCT"],params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

	# run raxml-ng for likelihood computation
	ll_rearr, rtime = return_likelihood(neighbor_tree_str, curr_path + MSA_PHYLIP_FILENAME, rates, pinv, alpha, freq)
