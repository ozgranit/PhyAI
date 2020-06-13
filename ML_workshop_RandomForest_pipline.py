import argparse
import sys

import numpy as np
import utils
sys.path.append("/groups/itay_mayrose/danaazouri/PhyAI/code/")
import warnings

warnings.filterwarnings("ignore")  # TEMP

#from defs import *

#from utils.tree_functions import get_total_branch_lengths
# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn import svm
# from sklearn.metrics import *
# from sklearn import preprocessing
# from sklearn.model_selection import cross_val_score
from statistics import mean, median
#from figures.violin_for_grant import *
#from figures.confidence_interval_dts import plot_pred_true
#from figures.accXsize_boxplot import accXsize_boxplot

##################
###### defs ######
LABEL = "d_ll_{}"
OPT_TYPE = "br"
KFOLD = 10  # "LOO"
GROUP_ID = 'group_id'
N_ESTIMATORS = 100
C = 95
FEATURE_SELECTION = False  # temp for running backwards selection


##################


def score_rank(df_by_ds, sortby, locatein, random, scale_score):
    '''
    find the best tree in 'sortby' (e.g., predicted as the best) foreach dataset and locate its rank in 'locatein' (e.g., y_test)
    '''
    best_pred_ix = df_by_ds[sortby].idxmax()  # changed min to max!
    if random:
        best_pred_ix = np.random.choice(df_by_ds[sortby].index, 1, replace=False)[0]
    temp_df = df_by_ds.sort_values(by=locatein, ascending=False).reset_index()  # changed ascending to False
    best_pred_rank = min(temp_df.index[temp_df["index"] == best_pred_ix].tolist())
    best_pred_rank += 1  # convert from pythonic index to position
    if scale_score:
        best_pred_rank /= len(df_by_ds[sortby].index)  # scale the rank according to rankmax

    return best_pred_rank


def get_cumsun_preds(df_by_ds):
    df_by_ds["pred"] /= df_by_ds["pred"].sum()
    assert round(df_by_ds["pred"].sum()) == 1
    temp_df = df_by_ds.sort_values(by="pred", ascending=False).reset_index()
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
    # print("***",threshold)
    # plt.hist(thresholds)
    # plt.show()
    for group_id, df_by_ds in grouped_df_by_ds:
        cumulative_scores = get_cumsun_preds(df_by_ds)["pred"].values
        res = round(100 * (len(np.where(cumulative_scores < threshold)[0])) / len(cumulative_scores), 2)
        cumsums.append(res)

    return cumsums


def ds_scores(df, move_type, random, scale_score):
    rank_pred_by_ds, rank_test_by_ds = {}, {}

    label = LABEL.format(move_type)
    sp_corrs, r2s, errs_down, errs_up, all_true, all_preds, thresholds = [], [], [], [], [], [], []
    grouped_df_by_ds = df.groupby(FEATURES[GROUP_ID], sort=False)
    for group_id, df_by_ds in grouped_df_by_ds:
        rank_pred_by_ds[group_id] = score_rank(df_by_ds, "pred", label, random, scale_score)
        rank_test_by_ds[group_id] = score_rank(df_by_ds, label, "pred", random, scale_score)

        all_true.append(df_by_ds[label].mean())
        pred = df_by_ds["pred"].mean()
        all_preds.append(pred)
        # r2s.append(r2_score(df_by_ds[label], df_by_ds["pred"], multioutput='variance_weighted'))
        temp_df = df_by_ds[[label, "pred"]]
        sp_corr = temp_df.corr(method='spearman').ix[1, 0]
        if sp_corr:
            if sp_corr < 0:
                print(group_id, sp_corr)
            sp_corrs.append(np.square(sp_corr))

        # compute 'confidence score'
        cumsum_true_best = get_cumsum_threshold(df_by_ds, label)
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

    regressor = RandomForestRegressor(n_estimators=N_ESTIMATORS, max_features=0.33, oob_score=True).fit(X_train,
                                                                                                        y_train)  # 0.33=nfeatures/3. this is like in R (instead of default=n_features)
    y_pred = regressor.predict(X_test)
    oob = regressor.oob_score_
    f_imp = regressor.feature_importances_

    all_DTs_pred = []
    # all_DTs_pred = [t.predict(X_test) for t in regressor.estimators_]
    # dev_vec = confidence_score(all_DTs_pred, y_pred, percentile=90)

    return y_pred, all_DTs_pred, oob, f_imp


def truncate(df):
    df = df.dropna()
    groups_ids = df[FEATURES[GROUP_ID]].unique()
    kfold = len(groups_ids) if KFOLD == "LOO" else KFOLD
    assert len(groups_ids) >= kfold
    ndel = len(groups_ids) % kfold
    if ndel != 0:  # i removed datasets from the end, and not randomly. from some reason..
        for group_id in groups_ids[:-ndel - 1:-1]:
            df = df[df[FEATURES[GROUP_ID]] != group_id]

    groups_ids = df[FEATURES[GROUP_ID]].unique()
    new_length = len(groups_ids) - ndel
    test_batch_size = int(new_length / kfold)

    return df.reset_index(drop=True), groups_ids, test_batch_size


def cross_validation_RF(df, move_type, features, validation_set=False, random=False, scale_score=True):
    # '''
    df, groups_ids, test_batch_size = truncate(df)
    res_dict = {}
    oobs, f_imps, = [], []
    my_y_pred, imps = np.full(len(df), np.nan), np.full(len(df), np.nan)

    if not validation_set:
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
            my_y_pred[
                df_test.index.values] = y_pred  # sort the predictions into a vector sorted according to the respective dataset

        df["pred"] = my_y_pred

    else:  # namely if validation set strategy, and not cross validation
        df_train = df
        df_test = pd.read_csv(dirpath + LEARNING_DATA.format("all_moves", "1_validation"))
        df_test = fit_transform(df_test, move_type, rank=False).dropna()
        y_pred, all_DTs_pred, oob, f_imp = apply_RFR(df_test, df_train, move_type, features)

        oobs.append(oob)
        f_imps.append(f_imp)
        df_test["pred"] = y_pred  # the predictions vec is the same lengths of test set
        df = df_test
    # '''

    rank_pred_by_ds, rank_test_by_ds, corrs, r2s, required_evaluations_scores = ds_scores(df, move_type, random,
                                                                                          scale_score)

    # averaged over cv iterations
    res_dict['oob'] = sum(oobs) / len(oobs)
    res_dict['f_importance'] = sum(f_imps) / len(f_imps)
    # foreach dataset (namely arrays are of lengths len(sampled_datasets)
    res_dict["rank_first_pred"] = rank_pred_by_ds
    res_dict["rank_first_true"] = rank_test_by_ds
    res_dict["spearman_corr"] = corrs
    res_dict['%neighbors'] = required_evaluations_scores

    groups_ids = df[FEATURES[GROUP_ID]].unique()
    suf = "_validation_set" if validation_set else ""
    df.to_csv(dirpath + DATA_WITH_PREDS.format(str(len(features)) + suf))  # + "_" + features[0]
    return res_dict, groups_ids


def fit_transform(df, move_type, rank=False):
    scores_range = (1, 100)
    groups_ids = df[FEATURES[GROUP_ID]].unique()
    for group_id in groups_ids:
        scaling_factor = df[df[FEATURES[GROUP_ID]] == group_id]["orig_ds_ll"].iloc[0]
        df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] /= scaling_factor
        if rank:
            df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)] = df.loc[
                df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)].rank(ascending=False)  # , pct=True)

    # MinMaxScaler
    # s = df[LABEL.format(move_type)].values
    # s_scaled = ((s - s.min(axis=0)) / (s.max(axis=0) - s.min(axis=0))) * (scores_range[1] - scores_range[0]) + scores_range[0]
    # df[LABEL.format(move_type)] = s_scaled

    # df.loc[df[FEATURES[GROUP_ID]] == group_id, LABEL.format(move_type)].plot.hist(by=FEATURES[GROUP_ID])
    # plt.show()

    return df


def parse_relevant_summaries_for_learning(df_orig, outpath, move_type, step_number, all_moves=False):
    ds_path_init = df_orig.iloc[0]["path"]
    cols = list(pd.read_csv(SUMMARY_PER_DS.format(ds_path_init, move_type, OPT_TYPE, step_number)))
    cols.insert(1, "path")
    cols.extend([FEATURES[GROUP_ID], FEATURES["group_tbl"]])  # add for group features
    df = pd.DataFrame(index=np.arange(0), columns=cols)

    for i, row in df_orig.iterrows():
        ds_path = row["path"]
        ds_tbl = get_total_branch_lengths(ds_path + PHYML_TREE_FILENAME.format('bionj'))
        summary_per_ds = SUMMARY_PER_DS.format(ds_path, move_type, OPT_TYPE, step_number)
        print(summary_per_ds)
        if os.path.exists(summary_per_ds) and FEATURES["bl"] in pd.read_csv(summary_per_ds).columns:
            df_ds = pd.read_csv(summary_per_ds)

            if all_moves:
                df_ds.insert(1, "path", ds_path)
                df_ds[FEATURES[GROUP_ID]] = str(i)
                df_ds[FEATURES["group_tbl"]] = ds_tbl
                df = pd.concat([df, df_ds], ignore_index=True)
            else:
                grouped = df_ds.groupby("{}_name".format(move_type), sort=False)
                for j, (name, group) in enumerate(grouped):
                    best_row_group = list(
                        group.ix[group[LABEL.format(move_type)].astype(float).idxmax()].values)  # changed min to max!
                    best_row_group.insert(1, ds_path)
                    best_row_group.extend([str(i), ds_tbl])  # add group features
                    df.ix[str(i) + "," + str(j)] = best_row_group

    df.to_csv(outpath)


def print_and_index_results(df_datasets, res_dict, move_type, sscore, features, val=False):
    #### score 1 ####
    spearman_corrs = res_dict['spearman_corr']
    df_datasets['corr'] = spearman_corrs
    print("\nsapearman corr:\n" + "mean:", mean([e for e in spearman_corrs if not math.isnan(e)]), ", median:",
          median(spearman_corrs))

    #### score 2 + 3 ####
    res_vec1 = np.asarray(list(res_dict['rank_first_pred'].values())) if type(res_dict['rank_first_pred']) is dict else \
        res_dict['rank_first_pred']
    res_vec2 = np.asarray(list(res_dict['rank_first_true'].values())) if type(res_dict['rank_first_true']) is dict else \
        res_dict['rank_first_true']
    scores_range = (1, 100)  # for MinMaxScaler
    res_vec1_scaled = ((res_vec1 - res_vec1.min(axis=0)) / (res_vec1.max(axis=0) - res_vec1.min(axis=0))) * (
            scores_range[1] - scores_range[0]) + scores_range[0]
    res_vec2_scaled = ((res_vec2 - res_vec2.min(axis=0)) / (res_vec2.max(axis=0) - res_vec2.min(axis=0))) * (
            scores_range[1] - scores_range[0]) + scores_range[0]
    df_datasets['best_predicted_ranking'] = res_vec1_scaled
    df_datasets['best_empirically_ranking'] = res_vec2_scaled
    print("\nbest predicted rank in true:\n" + "mean:", np.mean(res_vec1_scaled), ", median:",
          np.median(res_vec1_scaled))
    print("\nbest true rank in pred :\n" + "mean:", np.mean(res_vec2_scaled), ", median:", np.median(res_vec2_scaled))

    #### score 4 ####
    res_vec2_scaled.sort()
    sorted_true_scaled = res_vec2_scaled[::-1]
    index95 = int(0.05 * len(sorted_true_scaled) - 1)  # index for loc 0.05 = 95%
    required_evaluations = res_dict['%neighbors']
    df_datasets['required_evaluations_0.95'] = required_evaluations
    print("\nIn 0.95: {}%".format(sorted_true_scaled[index95]))
    print("\nmean %neighbors (0.95): {}".format(sum(required_evaluations) / len(required_evaluations)))

    # '''### feature importance ####
    mean_importances = res_dict['f_importance']  # index in first row only (score foreach run and not foreach dataset)
    for i, f in enumerate(features):
        colname = "imp_" + f
        df_datasets.loc[0, colname] = mean_importances[i]
    # print("\nmean f importance:\n", np.column_stack((features, mean_importances)))
	# plot_cumulative_importance(features, mean_importances, move_type, sscore)
    # '''
    #### additional information ####
    df_datasets.loc[0, 'oob'] = res_dict['oob']  # index in first row only (score foreach run and not foreach dataset)
    print("oob:", res_dict['oob'])
    print("ndatasets: ", len(res_vec1))

    suf = "_validation_set" if val else ""
    df_datasets.to_csv(dirpath + SCORES_PER_DS.format(str(len(features)) + suf))  # + "_" + features[0]
    print("##########################")

    return


def sort_features(res_dict, features):
    feature_importances = [(feature, round(importance, 4)) for feature, importance in
                           zip(features, res_dict['f_importance'])]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)  # most important first
    sorted_importances = [importance[1] for importance in feature_importances]
    sorted_features = [importance[0] for importance in feature_importances]

    return sorted_importances, sorted_features


def extract_scores_dict(res_dict, df_with_scores):
    ndatasets = len(df_with_scores)
    res_dict['rank_first_pred'], res_dict["rank_first_true"] = df_with_scores['best_predicted_ranking'].values, \
                                                               df_with_scores['best_empirically_ranking'].values
    res_dict['spearman_corr'], res_dict['%neighbors'], res_dict['oob'] = df_with_scores['corr'].values, df_with_scores[
        'required_evaluations_0.95'].values, df_with_scores.loc[0, 'oob']
    res_dict['f_importance'] = df_with_scores.loc[
        0, df_with_scores.columns[pd.Series(df_with_scores.columns).str.startswith('imp_')]].to_numpy()

    return res_dict, ndatasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arrange data for learning and implement learning algo')
    parser.add_argument('--move_type', '-mt', default='prune')  # could be 'prune' or 'rgft' or 'merged'
    parser.add_argument('--step_number', '-st', required=True)  # counting from 1
    parser.add_argument('--validation_set', '-val', default=False,
                        action='store_true')  # whether to use validation set INSTEAD of cross validation
    parser.add_argument('--all_moves', '-all', default=False,
                        action='store_true')  # necessary only if we want to learn rgft on all
    parser.add_argument('--rank_target', '-rank', default=False, action='store_true')
    parser.add_argument('--score_for_random', '-random', default=False, action='store_true')
    parser.add_argument('--scale_score', '-sscore', default=False, action='store_true')
    args = parser.parse_args()

    dirpath = SUMMARY_FILES_DIR if platform.system() == 'Linux' else DATA_PATH
    df_orig = pd.read_csv(dirpath + CHOSEN_DATASETS_FILENAME, dtype=types_dict)

    move_type = args.move_type
    ifrank = "" if not args.rank_target else "_rank"
    ifall = "" if not args.all_moves else "all_moves_"
    if not move_type == "merged":
        df_path = dirpath + LEARNING_DATA.format(ifall + move_type, str(args.step_number))
        if not os.path.exists(df_path):
            parse_relevant_summaries_for_learning(df_orig, df_path, move_type, args.step_number,
                                                  all_moves=args.all_moves)
    else:  # parse ALL neighbors to create a merged df off all features of all neighbors
        df_path = dirpath + LEARNING_DATA.format("all_moves", str(args.step_number))
        df_prune_features = dirpath + LEARNING_DATA.format("all_moves_prune", str(args.step_number))
        df_rgft_features = dirpath + LEARNING_DATA.format("all_moves_rgft", str(args.step_number))

        if not os.path.exists(df_path):
            parse_relevant_summaries_for_learning(df_orig, df_prune_features, "prune", args.step_number, all_moves=True)
            parse_relevant_summaries_for_learning(df_orig, df_rgft_features, "rgft", args.step_number, all_moves=True)
            shared_cols = FEATURES_SHARED + ["path", "prune_name", "rgft_name", "orig_ds_ll", "ll"]
            complete_df = pd.read_csv(df_prune_features, dtype=types_dict).merge(
                pd.read_csv(df_rgft_features, dtype=types_dict), on=shared_cols, left_index=True, right_index=True,
                suffixes=('_prune', '_rgft'))
            complete_df = complete_df.rename(columns={FEATURES[f]: FEATURES[f] + "_rgft" for f in FEATURES_RGFT_ONLY})
            complete_df[LABEL.format(move_type)] = complete_df[LABEL.format("prune")]
            complete_df.to_csv(df_path)

    df_learnig = pd.read_csv(df_path, dtype=types_dict)
    df_learnig = fit_transform(df_learnig, move_type, rank=args.rank_target)

    features = FEATURES_PRUNE if move_type == "prune" else FEATURES_RGFT if move_type == "rgft" else FEATURES_MERGED
    features.remove(FEATURES[GROUP_ID])
    features_to_drop = []

    ########################

    for i in range(len(features)):
        suf = "_validation_set" if args.validation_set else ""
        csv_with_scores = dirpath + SCORES_PER_DS.format(str(len(features)) + suf)
        if not os.path.exists(csv_with_scores) or args.validation_set:
            print("*@*@*@* scores for {} features are not available, thus applying learning".format(len(features)))
            res_dict, group_ids = cross_validation_RF(df_learnig, move_type, features,
                                                      validation_set=args.validation_set, random=args.score_for_random,
                                                      scale_score=args.scale_score)
            df_datasets = df_orig if not args.validation_set else pd.read_csv(
                DIRPATH + "/validation_set2/summary_files/" + CHOSEN_DATASETS_FILENAME)
        else:
            res_dict, ndatasets = extract_scores_dict({}, pd.read_csv(csv_with_scores, dtype=types_dict))
        print_and_index_results(df_datasets, res_dict, move_type, args.scale_score, features, args.validation_set)

        if not FEATURE_SELECTION:
            exit()
        else:
            sorted_importances, sorted_features = sort_features(res_dict, features)
            features = sorted_features[:-1]
            features_to_drop.append(sorted_features[-1])
            print("**dropped features:", features_to_drop)





