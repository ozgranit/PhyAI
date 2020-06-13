import sys
sys.path.append("/groups/itay_mayrose/danaazouri/PhyAI/code/")   ## CHANGEME
import warnings
warnings.filterwarnings("ignore")			# TEMP


from defs import *   ## CHANGEME
# from utils.tree_functions import get_total_branch_lengths   ## CHANGEME
from ML_workshop_utils import get_total_branch_lengths

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import *
from statistics import mean, median



OPT_TYPE = "br"
KFOLD = 10     # "LOO"
GROUP_ID = 'group_id'
N_ESTIMATORS = 70
C = 95
FIGURES = False

FIRST_ON_SEC = False  ## ignore!         # temp for running 1 on 2
SATURATION = False    ## ignore!          # temp to asses saturation

N_DATASETS = 100   # CHAMGEME!


def score_rank(df_by_ds, sortby, locatein, random=False, scale_score=False):
	'''
	find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
	'''

	best_pred_ix = df_by_ds[sortby].idxmax()    # changed min to max!
	temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()   # changed ascending to False
	best_pred_rank = min(temp_df.index[temp_df["index"] == best_pred_ix].tolist())
	best_pred_rank += 1  # convert from pythonic index to position
	
	if scale_score:
		best_pred_rank /= len(df_by_ds[sortby].index)   # scale the rank according to rankmax

	return best_pred_rank



def get_cumsun_preds(df_by_ds):
	df_by_ds["pred"] /= df_by_ds["pred"].sum()
	assert round(df_by_ds["pred"].sum()) == 1
	temp_df = df_by_ds.sort_values(by="pred", ascending=False).reset_index()      # ascending=True because the the sum() is negative so the order was changed
	sorted_preds = temp_df["pred"]
	cumsum_preds = sorted_preds.cumsum().values
	temp_df["pred"] = cumsum_preds
	
	return temp_df
	
	
def get_cumsum_threshold(df_by_ds, label):
	temp_df = get_cumsun_preds(df_by_ds)
	best_pred_ix = df_by_ds[label].idxmax()
	cumsum_true_best = temp_df[temp_df["index"] == best_pred_ix]["pred"].values[0]
	return cumsum_true_best


def calc_required_evaluations_score(grouped_df_by_ds, thresholds, c=C):
	cumsums = []
	threshold = np.percentile(thresholds, c)
	print("***",threshold)

	for group_id, df_by_ds in grouped_df_by_ds:
		cumulative_scores = get_cumsun_preds(df_by_ds)["pred"].values
		res = round(100 * (len(np.where(cumulative_scores < threshold)[0])) / len(cumulative_scores), 2)
		cumsums.append(res)

	return cumsums


def ds_scores(df, move_type, random, scale_score):
	rank_pred_by_ds, rank_test_by_ds = {}, {}

	label = LABEL.format(move_type)
	sp_corrs, r2s, errs_down, errs_up, all_true, all_preds, thresholds = [], [],[], [],[], [], []
	
	grouped_df_by_ds = df.groupby(FEATURES[GROUP_ID], sort=False)
	for group_id, df_by_ds in grouped_df_by_ds:
		# calc score 2 and 3
		rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", label, random, scale_score)
		rank_test_by_ds[group_id] = score_rank(df_by_ds, label, "pred", random, scale_score)

		# calc score 1
		temp_df = df_by_ds[[label, "pred"]]
		sp_corr = temp_df.corr(method='spearman').ix[1,0]
		if sp_corr:
			sp_corrs.append(sp_corr)
		else:
			sp_corrs.append(None)
		
		# calc score 4
		cumsum_true_best = get_cumsum_threshold(df_by_ds, label)
		if cumsum_true_best:
			thresholds.append(cumsum_true_best)
	required_evaluations_scores = calc_required_evaluations_score(grouped_df_by_ds, thresholds, c=C)


	return rank_pred_by_ds, rank_test_by_ds, sp_corrs, r2s, required_evaluations_scores


def split_features_label(df, move_type, features):
	attributes_df = df[features].reset_index(drop=True)
	label_df = df[LABEL.format(move_type)].reset_index(drop=True)

	x = np.array(attributes_df)
	y = np.array(label_df).ravel()

	return x, y


def apply_RFR(df_test, df_train, move_type, features):
	X_train, y_train = split_features_label(df_train, move_type, features)
	X_test, y_test = split_features_label(df_test, move_type, features)

	regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features=0.33,  oob_score=True).fit(X_train, y_train) # 0.33=nfeatures/3. this is like in R (instead of default=n_features)
	y_pred = regressor.predict(X_test)
	oob = regressor.oob_score_
	f_imp = regressor.feature_importances_

	all_DTs_pred = []


	return y_pred, all_DTs_pred, oob, f_imp


def truncate(df):  # removes nan rows if exist and split to folds
	df = df.dropna()
	groups_ids = df[FEATURES[GROUP_ID]].unique()

	selected_groups_ids = np.concatenate(np.random.choice(groups_ids, N_DATASETS, replace=False))
	df = df[df[FEATURES[GROUP_ID]].isin(selected_groups_ids)]
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	
	# very non elegant way to truncate to ndatasets that is divisible by KFOLD
	kfold = len(groups_ids) if KFOLD=="LOO" else KFOLD
	assert len(groups_ids) >= kfold
	ndel = len(groups_ids) % kfold
	if ndel != 0:   # i removed datasets from the end, and not randomly. from some reason..
		for group_id in groups_ids[:-ndel-1:-1]:
			df = df[df[FEATURES[GROUP_ID]] != group_id]

	
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	new_length = len(groups_ids) - ndel
	test_batch_size = int(new_length / kfold)

	return df.reset_index(drop=True), groups_ids, test_batch_size


def cross_validation_RF(df, move_type, features, trans=False, validation_set=False, random=False, scale_score=True):
	#'''
	df, groups_ids, test_batch_size = truncate(df)
	res_dict = {}
	oobs, f_imps, = [], []
	my_y_pred, imps = np.full(len(df), np.nan), np.full(len(df), np.nan)
	
	all_dts = [np.full(len(df), np.nan) for i in range(N_ESTIMATORS)]
	for low_i in groups_ids[::test_batch_size]:
		low_i, = np.where(groups_ids == low_i)
		low_i = int(low_i)
		up_i = low_i + test_batch_size

		test_ixs = groups_ids[low_i:up_i]
		train_ixs = np.setdiff1d(groups_ids, test_ixs)
		df_test = df.loc[df[FEATURES[GROUP_ID]].isin(test_ixs)]
		df_train = df.loc[df[FEATURES[GROUP_ID]].isin(train_ixs)]

		y_pred, all_DTs_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)

		oobs.append(oob)
		f_imps.append(f_imp)
		my_y_pred[df_test.index.values] = y_pred       # sort the predictions into a vector sorted according to the respective dataset

	df["pred"] = my_y_pred
	
	
	rank_pred_by_ds, rank_test_by_ds, corrs, r2s, required_evaluations_scores = ds_scores(df, move_type, random, scale_score)
	
	# averaged over cv iterations
	res_dict['oob'] = sum(oobs) / len(oobs)
	res_dict['f_importance'] = sum(f_imps) / len(f_imps)
	# foreach dataset (namely arrays are of lengths len(sampled_datasets)
	res_dict["rank_first_pred"] = rank_pred_by_ds
	res_dict["rank_first_true"] = rank_test_by_ds
	res_dict["spearman_corr"] = corrs
	res_dict['%neighbors'] = required_evaluations_scores
	
	
	return res_dict, df


def fit_transform(df, move_type, trans=False):
	groups_ids = df[FEATURES[GROUP_ID]].unique()
	for group_id in groups_ids:
		scaling_factor = df[df[FEATURES[GROUP_ID]] == group_id]["orig_ds_ll"].iloc[0]
		df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] /= -scaling_factor    # todo: make sure I run it with minus/abs to preserve order. also change 'ascending' to True in 'get_cumsun_preds' function

		if trans == 'rank':
			df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] = \
				df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)].rank(ascending=False) #, pct=True)
	if trans == "standard":
		scaler = StandardScaler()
		scaler.fit(df[LABEL.format(move_type)].values.reshape(-1,1))
		df[LABEL.format(move_type)] = scaler.transform(df[LABEL.format(move_type)].values.reshape(-1,1)) + 100
	if trans == 'exp':
		df[LABEL.format(move_type)] = np.exp2(df[LABEL.format(move_type)]+1)
		#df[LABEL.format(move_type)] = df[LABEL.format(move_type)].transform(np.exp2)
		
	
	##df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)].plot.hist(by=FEATURES[GROUP_ID])
	#df[LABEL.format(move_type)].plot.kde(by=FEATURES[GROUP_ID])
	#plt.show()

	return df



	
def print_and_index_results(df_datasets, res_dict, move_type, sscore, features):
	
	#### score 1 ####
	spearman_corrs = res_dict['spearman_corr']
	df_datasets['corr'] = spearman_corrs
	print("\nsapearman corr:\n" + "mean:", mean([e for e in spearman_corrs if not math.isnan(e)]), ", median:",median(spearman_corrs))
	
	#### score 2 + 3 ####
	res_vec1 = np.asarray(list(res_dict['rank_first_pred'].values())) if type(res_dict['rank_first_pred']) is dict else res_dict['rank_first_pred']
	res_vec2 = np.asarray(list(res_dict['rank_first_true'].values()))  if type(res_dict['rank_first_true']) is dict else res_dict['rank_first_true']
	#scores_range = (1, 100)   # for MinMaxScaler
	#res_vec1_scaled = ((res_vec1 - res_vec1.min(axis=0)) / (res_vec1.max(axis=0) - res_vec1.min(axis=0))) * (scores_range[1] - scores_range[0]) + scores_range[0]
	#res_vec2_scaled = ((res_vec2 - res_vec2.min(axis=0)) / (res_vec2.max(axis=0) - res_vec2.min(axis=0))) * (scores_range[1] - scores_range[0]) + scores_range[0]
	res_vec1_scaled = res_vec1
	res_vec2_scaled = res_vec2
	df_datasets['best_predicted_ranking'] = res_vec1_scaled
	df_datasets['best_empirically_ranking'] = res_vec2_scaled
	print("\nbest predicted rank in true:\n" + "mean:",np.mean(res_vec1_scaled), ", median:", np.median(res_vec1_scaled))
	print("\nbest true rank in pred :\n" + "mean:",np.mean(res_vec2_scaled), ", median:", np.median(res_vec2_scaled))
	
	#### score 4 ####
	res_vec2_scaled.sort()
	required_evaluations = res_dict['%neighbors']
	df_datasets['required_evaluations_0.95'] = required_evaluations
	print("\nmean %neighbors (0.95): {}".format(sum(required_evaluations)/len(required_evaluations)))
	
	#'''### feature importance ####
	mean_importances = res_dict['f_importance']   # index in first row only (score foreach run and not foreach dataset)
	for i, f in enumerate(features):
		colname = "imp_" + f
		df_datasets.loc[0, colname] = mean_importances[i]
	#print("\nmean f importance:\n", np.column_stack((features, mean_importances)))
	#plot_cumulative_importance(features, mean_importances, move_type, sscore)
	#'''
	#### additional information ####
	df_datasets.loc[0, 'oob'] = res_dict['oob']   # index in first row only (score foreach run and not foreach dataset)
	print("oob:", res_dict['oob'])
	print("ndatasets: ", len(res_vec1))
	
	print("##########################")
	return df_datasets



	




if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='arrange data for learning and implement learning algo')
	parser.add_argument('--move_type', '-mt', default='prune')	 # we will always work with move_type == "merged"
	parser.add_argument('--step_number', '-st', required=True) 	 # counting from 1
	parser.add_argument('--all_moves', '-all', default=False, action='store_true') # necessary only if we want to learn rgft on all
	parser.add_argument('--transform_target', '-trans', default=False)   # best transformation is "exp"
	parser.add_argument('--score_for_random', '-random', default=False, action='store_true')
	parser.add_argument('--scale_score', '-sscore', default=False, action='store_true')
	parser.add_argument('--tree_type', '-ttype', default='bionj')  # could be bionj or random
	args = parser.parse_args()

	dirpath = SUMMARY_FILES_DIR if platform.system() == 'Linux' else DATA_PATH
	df_orig = pd.read_csv(dirpath + CHOSEN_DATASETS_FILENAME, dtype=types_dict)

	move_type = args.move_type
	st = str(args.step_number)
	ifall = "" if not args.all_moves else "all_moves_"
	ifrandomstart = "" if args.tree_type == 'bionj' else "_random_starting"  # if == 'random'

	# parse ALL neighbors to create a merged df off all features of all neighbors
	df_path = dirpath + LEARNING_DATA.format("all_moves", st + ifrandomstart)
	df_learning = pd.read_csv(df_path, dtype=types_dict)
	df_learning = fit_transform(df_learning, move_type, trans=args.transform_target)
	
	features = FEATURES_PRUNE if move_type == "prune" else FEATURES_RGFT if move_type == "rgft" else FEATURES_MERGED
	features.remove(FEATURES[GROUP_ID])
	
	########################
	
	suf = "_{}_validation_set".format(st) if args.validation_set and not FIRST_ON_SEC else "_1st_on_2nd" if args.validation_set else "_{}".format(st)
	ifsaturation = "" if not SATURATION else "_" + str(N_DATASETS)
	ifrank = "" if not args.transform_target else "_ytransformed_{}".format(args.transform_target)
	ifrandomstart = "" if args.tree_type == 'bionj' else "_random_starting"   # if == 'random'
	suf += ifsaturation + ifrank + ifrandomstart
	
	csv_with_scores = dirpath + SCORES_PER_DS.format(str(len(features))+ suf)
	print("*@*@*@* scores for step{} with {} features are not available, thus applying learning".format(suf, len(features)))
	res_dict, df_out = cross_validation_RF(df_learning, move_type, features, trans=args.transform_target ,validation_set=False,random=False,scale_score=args.scale_score)
	df_out.to_csv(dirpath + DATA_WITH_PREDS.format(str(len(features)) + suf))
	
	df_datasets = df_orig
	df_datasets = df_datasets[df_datasets["path"].isin(df_out["path"].unique())]

	df_datasets = print_and_index_results(df_datasets, res_dict, move_type, args.scale_score, features)
	df_datasets.to_csv(csv_with_scores)