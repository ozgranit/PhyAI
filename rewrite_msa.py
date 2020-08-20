import re


def rewrite_in_phylip(msa_file):
	data = []
	with open(msa_file, 'r') as fp:
		for line in fp.readlines():
			nline = line.rstrip("\r\n")
			re_name_only = re.search("^(\S+)\s+\S+",nline)
			if re_name_only:
				name_only = re_name_only.group(1)
				end_name_ix = len(name_only) +1
				with_spaces = nline[:end_name_ix] + "      " + nline[end_name_ix:]
				nline = with_spaces
			data.append(nline)

	with open(msa_file, 'w') as nf:
		nf.write("\n".join(data))