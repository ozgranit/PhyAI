import os
import biopython as Bio
from Bio import AlignIO
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from ete3 import *

PHYLIP_FORMAT = "phylip-relaxed"


def get_msa_from_file(msa_file_path):
	if not os.path.exists(msa_file_path):
		return None
	try:
		msa = AlignIO.read(msa_file_path, PHYLIP_FORMAT)
	except:
		return None
	return msa


def get_msa_properties(msa):
	"""
	:param msa: bio.AlignIO format or path to msa file
	:return:
	"""
	if isinstance(msa, str):
		msa = get_msa_from_file(msa)
	ntaxa = len(msa)
	nchars = msa.get_alignment_length()

	return ntaxa, nchars


def get_seqs_dict(msa_file):
	alignment = AlignIO.read(msa_file, PHYLIP_FORMAT)
	seqs_dict = {seq.id: seq.seq for seq in alignment}
	return seqs_dict


def trunc_msa(subt, msa_path, trunc_msa_dest_path):
	# for calculating the intermediate subtrees created due to pruning or regrafting oerations
	'''
	:param subt: tree of newick format (str) or bio.AlignIO format
	:param msa_path: path to the msa containing all species in subt, and possibly more (usually the msa of the orig_tree from which subt is pruned)
	:param trunc_msa_dest_path: dest path to save the truncated msa
	:return: the truncated msa (according to the species exist in subt) in bio.AlignIO format
	'''
	if type(subt) == str:
		subt = Tree(newick=subt, format=1)
	records = []
	for leaf in subt.iter_leaves():  # go over all leaves (all species) in this subtree
		leaf_name = leaf.name
		seqs_dict = get_seqs_dict(msa_path)
		records.append(SeqRecord(seqs_dict[leaf_name], id=leaf_name))
	trunc_msa = MultipleSeqAlignment(records)
	AlignIO.write(trunc_msa, trunc_msa_dest_path, PHYLIP_FORMAT)

	return trunc_msa


def get_newick_tree(tree):
	"""
	:param tree: newick tree string or txt file containing one tree
	:return:	tree: a string of the tree in ete3.Tree format
	"""
	if os.path.exists(tree):
		with open(tree, 'r') as tree_fpr:
			tree = tree_fpr.read().strip()
	return tree


def get_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: list of branch lengths
	"""
	try:
		if type(tree) == str:
			tree = Tree(get_newick_tree(tree), format=1)
		tree_root = tree.get_tree_root()
	except:
		print(tree)
	if len(tree) == 1 and not "(" in tree:  # in one-branch trees, sometimes the newick string is without "(" and ")" so the .iter_decendants returns None
		return [tree.dist]
	branches = []
	for node in tree_root.iter_descendants(): # the root dist is 1.0, we don't want it
		branches.append(node.dist)

	return branches


def get_total_branch_lengths(tree):
	"""
	:param tree: Tree node or tree file or newick tree string;
	:return: total branch lengths
	"""
	branches = get_branch_lengths(tree)
	return sum(branches)