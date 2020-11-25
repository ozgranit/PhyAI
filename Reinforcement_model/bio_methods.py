import re, os, shutil
import pandas as pd
import networkx
from Bio import Phylo
from ete3 import Tree
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path

parent_path = Path().resolve().parent
parent_folder = parent_path / "reinforcement_data"

RAXML_NG_SCRIPT = "raxml-ng"  # after you install raxml-ng on your machine
# conda install -c bioconda raxml-ng
MSA_PHYLIP_FILENAME = "masked_species_real_msa.phy"


def return_likelihood(tree, msa_file, rates, pinv, alpha, freq):
	"""
	:param tree: ETEtree OR a newick string
	:param msa_file:
	:param rates: as extracted from parse_raxmlNG_content() returned dict
	:param pinv: as extracted from parse_raxmlNG_content() returned dict
	:param alpha: as extracted from parse_raxmlNG_content() returned dict
	:param freq: as extracted from parse_raxmlNG_content() returned dict
	:return: float. the score is the minus log-likelihood value of the tree
	"""
	model_line_params = 'GTR{rates}+I{pinv}+G{alpha}+F{freq}'.format(rates="{{{0}}}".format("/".join(rates)),
	                                                                 pinv="{{{0}}}".format(pinv),
	                                                                 alpha="{{{0}}}".format(alpha),
	                                                                 freq="{{{0}}}".format("/".join(freq)))

	# create tree file in memory and not in the storage:
	tree_rampath = "/dev/shm/" + msa_file.split("/")[-1] + "tree"  # the var is the str: tmp{dir_suffix}
	try:
		with open(tree_rampath, "w") as fpw:
			fpw.write(tree)

		p = Popen(
			[RAXML_NG_SCRIPT, '--evaluate', '--msa', msa_file, '--threads', '1', '--opt-branches', 'on', '--opt-model',
			 'off', '--model', model_line_params, '--nofiles', '--tree', tree_rampath],
			stdout=PIPE, stdin=PIPE, stderr=STDOUT)

		raxml_stdout = p.communicate()[0]
		raxml_output = raxml_stdout.decode()

		res_dict = parse_raxmlNG_content(raxml_output)
		ll = res_dict['ll']

	except Exception as e:
		print(msa_file.split("/")[-1])
		print(e)
		print("return_likelihood() in bio_methods.py")
		exit()
	finally:
		os.remove(tree_rampath)

	return float(ll)    # changed to return a num not a str


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
	if not ll_re and (
			re.search("BL opt converged to a worse likelihood score by", content) or re.search("failed", content)):
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
		for i, nuc in enumerate("ACGT"):
			res_dict["f" + nuc] = nucs_freq.group(i + 1).strip()

		# substitution frequencies
		subs_freq = re.search(
			"Substitution rates.*:\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)",
			content)
		for i, nuc_pair in enumerate(["AC", "AG", "AT", "CG", "CT", "GT"]):  # todo: make sure order
			res_dict["sub" + nuc_pair] = subs_freq.group(i + 1).strip()

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
	returns (a copy of) both ETE subtrees after pruning
	'''
	t_cp_p = t_orig.copy()  # the original tree is needed for each iteration
	assert t_cp_p & prune_name  # todo Oz: add indicative error
	prune_node_cp = t_cp_p & prune_name  # locate the node in the copied subtree
	assert prune_node_cp.up

	nname = prune_node_cp.up.name
	prune_loc = prune_node_cp
	prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
	t_cp_p.search_nodes(name=nname)[0].delete(
		preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

	return nname, prune_node_cp, t_cp_p


def regraft_branch(t_cp_p, prune_node_cp, rgft_name, nname):
	'''
	recieves: 2 ETE subtrees and 2 node names
	returns: an ETEtree with the 2 concatenated ETE subtrees
	'''

	t_temp = Tree()  # for concatenation of both subtrees ahead, to avoid polytomy
	t_temp.add_child(prune_node_cp)
	t_curr = t_cp_p.copy()
	assert t_curr & rgft_name  # todo Oz: add indicative error
	rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree
	new_branch_length = rgft_node_cp.dist / 2

	rgft_loc = rgft_node_cp.up
	rgft_node_cp.detach()
	t_temp.add_child(rgft_node_cp, dist=new_branch_length)
	t_temp.name = nname
	rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting

	return t_curr


def SPR_by_edge_names(ETEtree, cut_name, paste_name):
	nname, subtree1, subtree2 = prune_branch(ETEtree,
	                                         cut_name)  # subtree1 is the pruned subtree. subtree2 is the remaining subtree
	rearr_tree_str = regraft_branch(subtree2, subtree1, paste_name, nname).write(
		format=1)  # .write() is how you convert an ETEtree to newick string. now you can convert it back (if needed) using Tree(), or convert it to BIOtree

	return rearr_tree_str


def add_internal_names(tree_file, t_orig, newfile_suffix="_with_internal.txt"):
	# todo oz: I know you defined 'newfile_suffix' diferently (just None to runover?)
	# for tree with ntaxa=20 there are 2n-3 nodes --> n-3=17 internal nodes. plus one ROOT_LIKE node ==> always 18 internal nodes.
	N_lst = ["N{}".format(i) for i in range(1,19)]
	i = 0
	for node in t_orig.traverse():
		if not node.is_leaf():
			node.name = N_lst[i]
			i += 1
	# assuming tree file is a pathlib path
	new_tree_file = tree_file.parent / (tree_file.name + newfile_suffix)
	t_orig.write(format=3, outfile=new_tree_file)

	return t_orig, new_tree_file


# convert tree to weighted_adjacency_matrix
def tree_to_matrix(bio_tree):
	net = Phylo.to_networkx(bio_tree)
	matrix = networkx.adjacency_matrix(net)

	return matrix.toarray()


# returns the tree from the text file in the msa_num's folder
def get_tree_from_msa(msa_path="data/training_datasets/82/"):
	tree_path = parent_folder / (msa_path + "masked_species_real_msa.phy_phyml_tree_bionj.txt")

	with open(tree_path, "r") as f:
		tree_str = f.read()
	ete_tree = Tree(newick=tree_str, format=1)

	# add_internal_names does not run over the file it is given
	ete_tree, tree_copy = add_internal_names(tree_path, ete_tree)

	with open(tree_copy, "r") as f:
		tree_str = f.read()
	ete_tree.get_tree_root().name = "ROOT_LIKE"
	bio_tree = Phylo.read(tree_copy, "newick")
	os.remove(tree_copy)
	return ete_tree, bio_tree, tree_str


# converts string to other formats
def get_ete_and_bio_from_str(tree_str, msa_path):
	tree_copy = parent_folder / (msa_path + "for_now.txt")

	ete_tree = Tree(newick=tree_str, format=1)
	with open(tree_copy, "w") as f:
		f.write(tree_str)

	bio_tree = Phylo.read(tree_copy, "newick")
	os.remove(tree_copy)
	return ete_tree, bio_tree


# calculating likelihood of tree, msa_num should be the folder number of its corresponding msa
def get_likelihood_simple(tree_str, msa_path="data/training_datasets/82/", params=None):
	if params is None:
		# taking the params required for likelihood calculation from the stats file in the msa_num's folder
		freq, rates, pinv, alpha = calc_likelihood_params(msa_path)
	else:
		freq, rates, pinv, alpha = params

	msa_path = parent_folder / (msa_path + "masked_species_real_msa.phy")
	return return_likelihood(tree_str, str(msa_path), rates, pinv, alpha, freq)


def calc_likelihood_params(msa_path="data/training_datasets/82/"):
	stats_path = parent_folder / (msa_path + "masked_species_real_msa.phy_phyml_stats_bionj.txt")
	params_dict = parse_phyml_stats_output(stats_path)
	freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [
		params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"], params_dict["subCT"],
		params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

	return freq, rates, pinv, alpha


if __name__ == '__main__':
	# test on 1 dataset only
	DATAPATH = "/groups/itay_mayrose/danaazouri/PhyAI/ML_workshop/reinforcement_data/"
	df = pd.read_csv(DATAPATH + "data/sampled_datasets.csv")
	curr_path = DATAPATH + df.loc[0, "path"]
	tree_path = curr_path + MSA_PHYLIP_FILENAME + "_phyml_tree_bionj.txt"

	t_orig = Tree(newick=tree_path, format=1)  # ETEtree
	t_orig.get_tree_root().name = "ROOT_LIKE"  # ETEtree
	print(t_orig.get_ascii(show_internal=True))

	######## only for pre-processing ########
	ETEtree_with_internal_names, new_tree_path = add_internal_names(tree_path, t_orig)
	#########################################

	# a possible pair could NOT be something ELSE than all pair-combinations between [Sp000, ... , Sp0019] and [N1, ... , N18].  You can't know in advance which of THESE do exist (topology-dependant)
	neighbor_tree_str = SPR_by_edge_names(ETEtree_with_internal_names, 'Sp000', 'Sp002')

	# extract model params from the starting tree, to fix when calculating the likelihood of all neighbors
	starting_tree_path = curr_path + MSA_PHYLIP_FILENAME + "_phyml_stats_bionj.txt"
	params_dict = parse_phyml_stats_output(starting_tree_path)
	freq, rates, pinv, alpha = [params_dict["fA"], params_dict["fC"], params_dict["fG"], params_dict["fT"]], [
		params_dict["subAC"], params_dict["subAG"], params_dict["subAT"], params_dict["subCG"], params_dict["subCT"],
		params_dict["subGT"]], params_dict["pInv"], params_dict["gamma"]

	# run raxml-ng for likelihood computation
	ll_rearr = return_likelihood(neighbor_tree_str, curr_path + MSA_PHYLIP_FILENAME, rates, pinv, alpha, freq)
