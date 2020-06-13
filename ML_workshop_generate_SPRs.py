import shutil
from ete3 import *


##################
###### defs ######
SEP = "/"
MSA_PHYLIP_FILENAME = "real_msa.phy"
REARRANGEMENTS_DIRNAME = "rearrangements"
REARRANGEMENT_FILENAME = "rearrangement.txt"
ROOTLIKE_NAME = "ROOT_LIKE"
PHYML_TREE_FILENAME = MSA_PHYLIP_FILENAME + "_phyml_tree_{0}.txt"
##################


def prune_branch(t_orig, prune_name):
	'''
	get (a copy of) both subtrees after pruning
	'''
	t_cp_p = t_orig.copy()  				# the original tree is needed for each iteration
	prune_node_cp = t_cp_p & prune_name     # locate the node in the copied subtree
	assert prune_node_cp.up

	nname = prune_node_cp.up.name
	prune_loc = prune_node_cp
	prune_loc.detach()  # pruning: prune_node_cp is now the subtree we detached. t_cp_p is the one that was left behind
	t_cp_p.search_nodes(name=nname)[0].delete(preserve_branch_length=True)  # delete the specific node (without its childs) since after pruning this branch should not be divided

	return nname, prune_node_cp, t_cp_p


def regraft_branch(t_cp_p, rgft_node, prune_node_cp, rgft_name, nname):
	'''
	get a tree with the 2 concatenated subtrees
	'''
	new_branch_length = rgft_node.dist /2   # to assign a new (arbitrary) length to the regrafted branch
	t_temp = PhyloTree()  			   # for concatenation of both subtrees ahead, to avoid polytomy
	t_temp.add_child(prune_node_cp)
	t_curr = t_cp_p.copy()
	rgft_node_cp = t_curr & rgft_name  # locate the node in the copied subtree

	rgft_loc = rgft_node_cp.up
	rgft_node_cp.detach()
	t_temp.add_child(rgft_node_cp, dist=new_branch_length)
	t_temp.name = nname
	rgft_loc.add_child(t_temp, dist=new_branch_length)  # regrafting

	return t_curr


def save_rearr_file(trees_dirpath, rearrtree, filename):
	if not os.path.exists(trees_dirpath):
		os.makedirs(trees_dirpath)
	tree_path = trees_dirpath + filename
	if not os.path.exists(tree_path):
		rearrtree.write(format=1, outfile=tree_path)

	return tree_path


def add_internal_names(tree_file, tree_file_cp_no_internal, t_orig):
	shutil.copy(tree_file, tree_file_cp_no_internal)
	for i, node in enumerate(t_orig.traverse()):
		if not node.is_leaf():
			node.name = "N{}".format(i)
	t_orig.write(format=3, outfile=tree_file)   # runover the orig file with no internal nodes names
	
	
def get_tree(ds_path, msa_file):
	tree_file = ds_path + PHYML_TREE_FILENAME.format("bionj")
	tree_file_cp_no_internal = ds_path + PHYML_TREE_FILENAME.format("bionj_no_internal")
	if not os.path.exists(tree_file_cp_no_internal):
		t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=1)
		add_internal_names(tree_file, tree_file_cp_no_internal, t_orig)
	else:
		t_orig = PhyloTree(newick=tree_file, alignment=msa_file, alg_format="iphylip", format=3)

	return t_orig


def all_SPR(ds_path, tree=None):
	orig_msa_file = ds_path + MSA_PHYLIP_FILENAME
	t_orig = get_tree(ds_path, orig_msa_file) if not tree else PhyloTree(newick=tree, alignment=orig_msa_file, alg_format="iphylip", format=1)
	t_orig.get_tree_root().name = ROOTLIKE_NAME
	for i, prune_node in enumerate(t_orig.iter_descendants("levelorder")):
		prune_name = prune_node.name
		nname, subtree1, subtree2 = prune_branch(t_orig, prune_name) # subtree1 is the pruned subtree. subtree2 is the remaining subtree
		subtrees_dirpath = SEP.join([ds_path, REARRANGEMENTS_DIRNAME, prune_name, "{}", ""])

		for j, rgft_node in enumerate(subtree2.iter_descendants("levelorder")):
			rgft_name = rgft_node.name
			if nname == rgft_name: # if the rgrft node is the one that was pruned
				continue

			full_tree_dirpath = subtrees_dirpath.format(rgft_name)
			if not os.path.exists(full_tree_dirpath + REARRANGEMENT_FILENAME):
				full_tree = regraft_branch(subtree2, rgft_node, subtree1, rgft_name, nname)
				save_rearr_file(full_tree_dirpath, full_tree, filename=REARRANGEMENT_FILENAME)
				# call phyml to calc likelihood(full_tree)
	return





if __name__ == '__main__':
	dirpath = "<the path to the dir containing an MSA and a tree file>"   # for example "Drosophila_00050000000014001/"
	# use tree argument if you wish to give a newick string as an input (instead of dirpath containing the two files)
	all_SPR(dirpath, tree=None)