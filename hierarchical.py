import numpy as np

# import load_data as ld
import sys
from evaluation import report
import os
import json
import scipy.io
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import argparse

# import plots
import utils
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import normalize


# Computes a single observation matrix and returns it
def compute_obs_matrix(enum, Z, obs_matrix_step):
    i, j = enum
    row = Z[j]
    print("computing matrix", i * obs_matrix_step + 1, "out of", len(Z))
    dist = row[2]
    labels_pred = fcluster(Z, dist, criterion="distance")
    obs_m = utils.build_observation_matrix(labels_pred)
    return obs_m


def hierarchical_judgement_matrix(x, y, obs_matrix_step):
    print("Building observation matrices")
    # sort data by labels to improve the obs matrix
    sorted_idx = y.argsort()
    x = x[sorted_idx]
    y = y[sorted_idx]

    judgement_matrix = np.zeros((y.shape[0], y.shape[0]))
    # Z = linkage(x, 'average', metric='cosine')

    # quick test. Fix later
    x = normalize(x)
    Z = linkage(x, "ward", metric="euclidean")
    ####

    compute_obs_matrix_p = partial(
        compute_obs_matrix, Z=Z, obs_matrix_step=obs_matrix_step
    )
    # Creates pool of workers to compute all observation matrices in parallel and accumulates partial results on judgement_matrix
    with Pool() as pool:
        for obs_m in pool.imap_unordered(
            compute_obs_matrix_p, enumerate(range(0, len(Z), obs_matrix_step))
        ):
            judgement_matrix += obs_m

    return judgement_matrix


def hierarchical(
    x, n_clusters, affinity, linkage, distance_threshold, normalize_data=False
):
    if normalize_data:
        x = normalize(x.copy())
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        linkage=linkage,
        distance_threshold=None,
    )
    # print('\nPerforming Agglomerative clustering with k =',n_clusters)
    predictions = hierarchical.fit_predict(x)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform clustering using hierarchical agglomerative clustering"
    )
    parser.add_argument(
        "--data",
        type=str,
        help="path to data in npy format.",
        default="/home/users/matheus/doutorado/paper_open_video/final_results/ucf101/train_2__test_6__tail_10__cover_0.1__seed_42/0/triplet/x_test_features.npy",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="path to labels in npy format",
        default="/home/users/matheus/doutorado/paper_open_video/final_results/ucf101/train_2__test_6__tail_10__cover_0.1__seed_42/0/triplet/y_test.npy",
    )
    parser.add_argument(
        "--opensetlabels",
        type=str,
        help="path to npy array with ground truth known/unknown labels (used to remove all knowns from the dataset)",
        default="/home/users/matheus/doutorado/paper_open_video/final_results/ucf101/train_2__test_6__tail_10__cover_0.1__seed_42/0/triplet/open_y_test.npy",
    )
    parser.add_argument("--k", type=int, help="number of clusters.")
    parser.add_argument(
        "--affinity",
        type=str,
        help="Metric used to compute the linkage. Default Cosine.",
        default="cosine",
    )
    parser.add_argument(
        "--linkage", type=str, help="Linkage type. Default complete.", default="average"
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        help="Distance threshold above which clusters will not be merged. Overrides K.",
        default=-9999.0,
    )
    parser.add_argument(
        "--small_dendrogram", help="Output a small dendrogram.", action="store_true"
    )
    parser.add_argument(
        "--large_dendrogram", help="Output a large dendrogram.", action="store_true"
    )
    parser.add_argument(
        "--a_priori",
        help="Use a priori knowledge about the data (known classes) to determine distance threshold. Overrides distance_threshold",
        action="store_true",
    )
    parser.add_argument(
        "--obs_matrix",
        help="Generate observation matrixes at every cut of the dendrogram",
        action="store_true",
    )
    parser.add_argument(
        "--obs_matrix_step",
        help="Step size when generating the observation matrix",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output_path",
        type=str,
        help="where to output results",
        default="results/hierarchical/2_6_0_wardtest",
    )

    args = parser.parse_args()
    print(args)

    if args.output_path[-1] != "/":
        args.output_path += "/"

    # insert a function which checks if this algorithm with this parameters has already been executed and skip

    # load data and labels
    x, y = ld.load(args.data, args.labels, args.opensetlabels)

    if args.linkage == "ward":
        from sklearn.preprocessing import normalize

        x = normalize(x)

    if args.obs_matrix:
        judgement_matrix = hierarchical_judgement_matrix(x, y, args.obs_matrix_step)
        plots.plot_cov_matrix(
            judgement_matrix,
            args.output_path,
            "judgement_matrix.png",
            title="judgement matrix",
        )

        ground_truth_obs = utils.build_observation_matrix(y)
        plots.plot_cov_matrix(
            ground_truth_obs,
            args.output_path,
            "ground_truth_observation_matrix.png",
            title="ground truth matrix",
        )

    if args.a_priori:
        print("\nSearching for distance threshold that best fits known data...")
        Z = linkage(x, "average", metric="cosine")
        best = 0
        for row in Z:
            dist = row[2]
            labels_pred = fcluster(Z, dist, criterion="distance")
            ami = metrics.adjusted_mutual_info_score(y, labels_pred)
            if ami > best:
                best = ami
                args.distance_threshold = dist

    if args.distance_threshold != -9999.0:
        hierarchical = AgglomerativeClustering(
            n_clusters=None,
            affinity=args.affinity,
            linkage=args.linkage,
            distance_threshold=args.distance_threshold,
        )
        print(
            "\nPerforming Agglomerative clustering with distance threshold =",
            args.distance_threshold,
        )

    else:
        assert args.k is not None, "define a value for K"
        hierarchical = AgglomerativeClustering(
            n_clusters=args.k,
            affinity=args.affinity,
            linkage=args.linkage,
            distance_threshold=None,
        )
        print("\nPerforming Agglomerative clustering with k =", args.k)

    predictions = hierarchical.fit_predict(x)
    report(
        x, y, predictions, args.output_path, distance_threshold=args.distance_threshold
    )

    # plot dendrograms
    if args.small_dendrogram or args.large_dendrogram:
        if args.distance_threshold == -9999.0:
            hierarchical = AgglomerativeClustering(
                n_clusters=None,
                affinity=args.affinity,
                linkage=args.linkage,
                distance_threshold=0.0,
            ).fit(x)
        if args.small_dendrogram:
            print("\nPlotting small dendrogram...")
            plots.plot_dendrogram(
                hierarchical, args.output_path, labels=y, large_plot=False
            )
        if args.large_dendrogram:
            print("\nPlotting large dendrogram...")
            plots.plot_dendrogram(
                hierarchical, args.output_path, labels=y, large_plot=True
            )
