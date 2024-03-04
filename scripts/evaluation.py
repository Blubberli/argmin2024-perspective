import argparse
from collections import Counter

import numpy as np
import pandas as pd
import tabulate
import os
from tqdm import tqdm
from evaluate_diversity import get_perspectives, alpha_ndcg, calculateNormalizedDiscountedKLDivergence
from evaluate_relevance import ndcg, precision_at_k
from utils import read_gold_data

political_issues = {'Liberale Gesellschaft', 'Ausgebauter Umweltschutz', 'Restriktive Finanzpolitik', 'Law & Order',
                    'Liberale Wirtschaftspolitik', 'Restriktive Migrationspolitik', 'Ausgebauter Sozialstaat'}

political_issues_translations = {
    'Liberale Gesellschaft': 'Liberal Society',
    'Ausgebauter Umweltschutz': 'Enhanced Environmental Protection',
    'Restriktive Finanzpolitik': 'Restrictive Fiscal Policy',
    'Law & Order': 'Law & Order',
    'Liberale Wirtschaftspolitik': 'Liberal Economic Policy',
    'Restriktive Migrationspolitik': 'Restrictive Immigration Policy',
    'Ausgebauter Sozialstaat': 'Expanded Welfare State'
}


def get_corpus_perspectives(corpus, sociovars):
    """
    For the corpus and each socio-cultural variable, create a list of perspectives for each argument_id. The list of
    perspectives is a list of the values of the socio-cultural variable for each argument_id, e.g. for the variable
    'residence' the list would be ['urban', 'rural', 'urban', 'urban', 'rural', ...]
    :param corpus: the corpus dataframe
    :param sociovars: a set of socio-cultural variables, e.g. age, gender etc.
    :return: a dictionary with the socio-cultural variable as key and a dictionary of argument_id to concrete value of
    the socio-cultural variable
    """
    all_argument_ids = list(corpus.index)
    variable2perspectives = {}
    for variable in sociovars:
        if variable == "age_bin":
            variable = "age"
            perspectives = get_perspectives(all_argument_ids, corpus, variable)
        elif variable == "important_political_issue":
            for issue in political_issues:
                variable = issue
            perspectives = get_perspectives(all_argument_ids, corpus, variable)
        else:
            perspectives = get_perspectives(all_argument_ids, corpus, variable)
        # zip as a dictionary of argument_id to perspective
        perspectiveDict = dict(zip(all_argument_ids, perspectives))
        variable2perspectives[variable] = perspectiveDict
    return variable2perspectives


def evaluate_relevance(predictions_df, ground_truth_df, output_dir):
    """
    Evaluates the relevance of ranking predictions by comparing them with the ground truth data.
    The evaluation involves computing the Normalized Discounted Cumulative Gain (NDCG) and
    precision at k (precision@k) metrics for each query ID at predefined k.

    Parameters:
    - predictions_df (pandas.DataFrame): A DataFrame containing the predictions for each query.
      It must include at least two columns: 'query_id', which uniquely identifies each query, and
      'candidates', which contains lists of the top 1000 ranked candidate items (argument_ids)
      predicted for each query.

    - ground_truth_df (pandas.DataFrame): A DataFrame containing the ground truth data for
      each query. Similar to `predictions_df`, it should include 'query_id' and 'candidates'
      columns. Here, 'candidates' contains lists of all relevant candidate items (argument_ids)
      for each query, as determined by ground truth annotations.

    - corpus (pandas.DataFrame): The DataFrame containing detailed information about each item
      (argument) in the corpus. This includes 'argument_id' and various socio-cultural variables
      associated with each argument.
    """
    predicted_candidates = predictions_df["relevant_candidates"].values
    ground_truth_candidates = ground_truth_df["relevant_candidates"].values
    ground_truth_sets = [set(gt) for gt in ground_truth_candidates]
    k_range = [4, 8, 16, 20]
    results_relevance = []
    for i in tqdm(range(len(predictions_df))):
        query_id = ground_truth_df.iloc[i]["query_id"]
        predicted = predicted_candidates[i]
        ground_truth_set = ground_truth_sets[i]
        ground_truth = ground_truth_candidates[i]
        relevance_score = [1 if candidate in ground_truth_set else 0 for candidate in predicted]
        if len(ground_truth) < 1000:
            sample_size = 1000 - len(ground_truth)
            relevance_scores_gold = [1] * len(ground_truth_set) + [0] * sample_size
        else:
            relevance_scores_gold = [1] * 1000
        results_ndcg = ndcg(relevance_score, relevance_scores_gold, k_range)
        for k in k_range:
            precisionk = precision_at_k(relevance_score, k)
            results_relevance.append({
                "query_id": query_id,
                "k": k,
                "ndcg@k": results_ndcg[k],
                "precision@k": precisionk,
            })
    results_relevance_df = pd.DataFrame(results_relevance)
    # aggregate mean over query_id
    results_relevance_df = results_relevance_df.groupby("k").mean().reset_index()
    # drop query_id
    results_relevance_df = results_relevance_df.drop(columns=["query_id"])
    # trec_eval.compute_batch()
    print(tabulate.tabulate(results_relevance_df, headers="keys", tablefmt="psql"))
    results_relevance_df.to_csv(f"{output_dir}/relevance_results.csv", index=False)


def evaluate_diversity(predictions_df, ground_truth_df, corpus, output_dir):
    """
    Evaluates the diversity of ranking predictions by comparing them against the ground truth data.
    This evaluation is conducted by computing two metrics for each query ID and specified ranks k:
    the alpha-Normalized Discounted Cumulative Gain (alpha-NDCG) and the Normalized discounted KL-divergence (rKL)
    divergence.

    Parameters:
    - predictions_df (pandas.DataFrame): A DataFrame containing the predictions for each query. It
      should have at least two columns: 'query_id' and 'candidates'. The 'candidates' column must
      contain lists of the top 1000 ranked candidates (items) predicted for each query.

    - ground_truth_df (pandas.DataFrame): A DataFrame containing the ground truth data for each query.
      Similar to `predictions_df`, it should have at least two columns: 'query_id' and 'candidates'.
      Here, the 'candidates' column contains lists of all relevant candidates for each query, as determined
      by ground truth annotations.

    - corpus (pandas.DataFrame): The corpus DataFrame that includes detailed information about each candidate
      or item that may appear in the rankings.

    - output_dir (str): The path to the output directory where the results of the diversity evaluation
      will be saved.
    """
    predicted_candidates = predictions_df["relevant_candidates"].values
    ground_truth_candidates = ground_truth_df["relevant_candidates"].values
    ground_truth_sets = [set(gt) for gt in ground_truth_candidates]
    argument_ids = set(corpus.index.values)
    socio_vars = set(corpus.columns).difference({"topic", "argument", "denomination", "important_political_issues"})
    variable2perspectives = get_corpus_perspectives(corpus, socio_vars)
    k_range = [4, 8, 16, 20]
    cut_off_points = [2, 4, 8, 10, 12, 14, 16, 18, 20]
    results_diversity = []
    for i in tqdm(range(len(predictions_df))):
        query_id = ground_truth_df.iloc[i]["query_id"]
        predicted = predicted_candidates[i]
        ground_truth_set = ground_truth_sets[i]
        ground_truth = ground_truth_candidates[i]
        relevance_score = [1 if candidate in ground_truth_set else 0 for candidate in predicted]
        if len(ground_truth) < 1000:
            sample_size = 1000 - len(ground_truth)
            argument_ids_not_in_ground_truth = argument_ids - ground_truth_set
            sample_ids = np.random.choice(list(argument_ids_not_in_ground_truth), sample_size, replace=False)
            relevance_scores_gold = [1] * len(ground_truth_set) + [0] * sample_size
            ground_truth.extend(sample_ids)
        else:
            relevance_scores_gold = [1] * 1000
            # cut ground truth to 1000
            ground_truth = ground_truth[:1000]
        for variable in socio_vars:
            perspectives_dict = variable2perspectives[variable]
            # use Counter to get the relative amount of each perspective in the ground truth
            gold_distribution = Counter(list(perspectives_dict.values()))
            # convert each count into a proportion
            total = sum(gold_distribution.values())

            # remove the majority class from the gold distribution
            gold_distribution.pop(max(gold_distribution, key=gold_distribution.get))
            relative_gold_distribution = {k: v / total for k, v in gold_distribution.items()}
            # lookup perspectives of ground_truth based on argument_ids in the list of ground_truth
            perspectives_ground_truth = [perspectives_dict[argument_id] for argument_id in ground_truth]
            perspectives_predictions = [perspectives_dict[argument_id] for argument_id in predicted]

            alpha_ndcg_score = alpha_ndcg(relevance_scores_predictions=relevance_score,
                                          perspectives_predictions=perspectives_predictions,
                                          relevance_scores_global=relevance_scores_gold,
                                          perspectives_global=perspectives_ground_truth, alpha=0.5, k_range=k_range)
            for k in k_range:
                kl_socio_var = get_kl_divergence(k=k, cutoff_points=cut_off_points,
                                                 gold_distribution=relative_gold_distribution,
                                                 ranked_perspectives=perspectives_predictions)
                results_diversity.append({
                    "query_id": query_id,
                    "k": k,
                    "socio_var": variable,
                    "alpha_ndcg@k": alpha_ndcg_score[k],
                    "kl_divergence@k": kl_socio_var
                })
    results_diversity_df_fine = pd.DataFrame(results_diversity).groupby(["k", "socio_var"]).mean().reset_index()
    results_diversity_df_fine["socio_var"] = results_diversity_df_fine["socio_var"].apply(
        lambda x: political_issues_translations[x] if x in political_issues_translations else x)
    results_diversity_df_fine = results_diversity_df_fine.drop(columns=["query_id"])
    # save the results to a csv file
    results_diversity_df_fine.to_csv(f"{output_dir}/diversity_results_per_variable.csv", index=False)
    # aggregate mean over socio_var
    # drop socio_var
    results_diversity_df = results_diversity_df_fine.drop(columns=["socio_var"])
    results_diversity_df = results_diversity_df.groupby("k").mean().reset_index()
    print(tabulate.tabulate(results_diversity_df, headers="keys", tablefmt="psql"))
    results_diversity_df.to_csv(f"{output_dir}/diversity_results.csv", index=False)


def get_kl_divergence(k, cutoff_points, gold_distribution, ranked_perspectives):
    klds_for_variable = []
    for socio_value, count in gold_distribution.items():
        kld = calculateNormalizedDiscountedKLDivergence(ranked_perspectives=ranked_perspectives, cut_off_points=cutoff_points,
                                                        k=k, protected_group=socio_value, gold_propotion=count)
        klds_for_variable.append(kld)
    return sum(klds_for_variable) / len(klds_for_variable)


if __name__ == '__main__':
    # read in command line arguments
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--data", type=str, required=True, help="Path to the data directory")
    argument_parser.add_argument("--scenario", type=str, required=True,  help="Which scenario do you want to evaluate? Either 'baseline' for scenario 1 or 'perspective' for scenario 2 and 3")
    argument_parser.add_argument("--split", type=str, required=True, help="Which split do you want to evaluate? Either 'train' or 'dev'")
    argument_parser.add_argument("--predictions", type=str, required=True, help="Path to the predictions file")
    argument_parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    argument_parser.add_argument("--diversity", type=bool, required=False, default=False, help="Evaluate diversity")
    args = argument_parser.parse_args()

    scenario = args.scenario
    split = args.split

    data = read_gold_data(args.data)
    corpus = data["corpus"]
    ground_truth = data[scenario][split]
    # sort corpus df by argument_id
    corpus = corpus.sort_values("argument_id")
    # set index to argument_id
    corpus = corpus.set_index("argument_id")
    # drop column demographic_profile for evaluation
    corpus = corpus.drop(columns=["demographic_profile"])

    predictions = pd.read_json(args.predictions, lines=True, orient="records")
    predictions = predictions.sort_values("query_id")

    output_dir = f"{args.output_dir}/{scenario}/{split}"

    os.system(f"mkdir -p {output_dir}")

    evaluate_relevance(ground_truth_df=ground_truth, predictions_df=predictions, output_dir=output_dir)
    if args.diversity:
        evaluate_diversity(ground_truth_df=ground_truth, predictions_df=predictions, corpus=corpus, output_dir=output_dir)
