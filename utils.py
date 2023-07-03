import os
import numpy as np


def train_test_split_groups(filenames, initial_classes, params):
    videos = [x.split(".")[0] for x in filenames if x.split("/")[0] in initial_classes]
    # print(videos)

    merged_list = (
        []
    )  # this list is for merging groups of the same class so that we can use sklearns train_test_split method. Each element in the list is a group of videos that belong to the same class and group
    for cl in initial_classes:
        videos_current_class = sorted([x for x in videos if x.split("/")[0] == cl])
        groups_current_class = np.unique(
            [x.split("_")[2] for x in videos_current_class]
        )
        for g in groups_current_class:
            videos_group_current_class = sorted(
                [x for x in videos_current_class if x.split("_")[2] == g]
            )
            merged_list.append(videos_group_current_class)

    merged_list_labels = [x[0].split("/")[0] for x in merged_list]

    # employ sklearns train test split method to generate train/test knowns
    from sklearn.model_selection import train_test_split

    (
        merged_knowns_train,
        merged_knowns_test,
        merged_knowns_train_labels,
        merged_knowns_test_labels,
    ) = train_test_split(
        merged_list,
        merged_list_labels,
        train_size=0.7,
        random_state=params["seed"],
        shuffle=True,
        stratify=merged_list_labels,
    )

    # unmerge lists
    train = []
    train_labels = []
    for m, l in zip(merged_knowns_train, merged_knowns_train_labels):
        for n in m:
            train.append(n)
            train_labels.append(l)

    test = []
    test_labels = []
    for m, l in zip(merged_knowns_test, merged_knowns_test_labels):
        for n in m:
            test.append(n)
            test_labels.append(l)

    print(initial_classes, merged_list)

    return train, train_labels, test, test_labels


def train_test_split_hmdb51(
    PATH_TO_SPLITS, train_file, test_file, unique_classes, params
):
    with open(PATH_TO_SPLITS + train_file) as f:
        train = [line.rstrip().split(".")[0] for line in f]
        train_labels = [line.split("/")[0] for line in train]
        print(train_labels, train[-1])
    with open(PATH_TO_SPLITS + test_file) as f:
        test = [line.rstrip().split(".")[0] for line in f]
        test_labels = [line.split("/")[0] for line in test]

    return train, train_labels, test, test_labels


def map_labels(classes):
    dict_map = {}
    for i, c in enumerate(sorted(classes)):
        dict_map[c] = i + 1
    for key, value in sorted(dict_map.items()):
        print(key, value)
    return dict_map


def convert_labels_to_int(labels, dict_map):
    new_labels = []
    for l in labels:
        new_labels.append(dict_map[l])
    return new_labels


def openness(training_classes, testing_classes, target_classes):
    from math import sqrt

    return 1 - (sqrt((2.0 * training_classes) / (testing_classes + target_classes)))


def openness_Geng(training_classes, testing_classes):
    from math import sqrt

    return 1 - (sqrt((2.0 * training_classes) / (testing_classes + training_classes)))


def gen_exp_id(params):
    exp_id = params["model"] + "/"
    exp_id += (
        "train_"
        + str(params["n_train_classes"])
        + "__test_"
        + str(params["n_test_classes"])
        + "__"
    )
    # exp_id += 'tail_' + str(params['tail_size']) + '__cover_' + str(params['cover_threshold']) + '__'
    exp_id += "tail_" + str(10) + "__cover_" + str(params["cover_threshold"]) + "__"

    exp_id += "seed_" + str(params["init_seed"]) + "/"

    return exp_id


def makedirs(path):
    try:
        os.makedirs(path)
    except Exception as e:
        print(e)


def generate_report(youdens_index, closed_f1_score, classif_rep, cm, params):
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"
    output_path += str(params["classification_threshold"]) + "/"
    makedirs(output_path)

    outfile = open(output_path + "classification_report.txt", "w")

    outfile.write("problem openness: " + str(params["openness"]) + "\n")
    outfile.write(str(classif_rep) + "\n")
    outfile.write("\n" + str(cm) + "\n")
    outfile.write("\nopen youdens index: " + str(youdens_index) + "\n")
    outfile.write("closed f1 score:" + str(closed_f1_score) + "\n")

    outfile.close()


def generate_clustering_report(x, y, pred, params, **kwargs):
    import evaluation

    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"
    output_path += str(params["classification_threshold"]) + "/"
    makedirs(output_path)

    results = evaluation.clustering_metrics(x, y, pred)

    f = open(output_path + "clustering_report.txt", "w")

    f.write("Ground truth number of clusters = " + str(results["n_clusters_gt"]) + "\n")
    f.write("Number of clusters found = " + str(results["k"]) + "\n")

    # check kwargs for additional info
    if kwargs.get("additional_info") is not None:
        dict = kwargs.get("additional_info")
        for key, value in dict.items():
            print(key, value)
            f.write(str(key) + ": " + str(value) + "\n")

    f.write("\nHomogeneity: " + str(results["homogeneity"]) + "\n")
    f.write("Completeness: " + str(results["completeness"]) + "\n")
    f.write("V-measure: " + str(results["vmeasure"]) + "\n")
    f.write("Adjusted Rand score: " + str(results["adjusted_rand"]) + "\n")
    f.write("Adjusted mutual info: " + str(results["adjusted_mutual_info"]) + "\n")
    f.write("Calinski Harabasz score: " + str(results["calinski_harabasz"]) + "\n")
    f.write("Davies Bouldin score: " + str(results["davies_bouldin"]) + "\n")
    f.write("Fowlkes Mallows score: " + str(results["fowlkes_mallows"]) + "\n")
    f.write("Mutual Info score: " + str(results["mutual_info"]) + "\n")
    f.write(
        "Normalized Mutual Info score: " + str(results["normalized_mutual_info"]) + "\n"
    )
    f.write(
        "Silhouette score (Euclidean distance): "
        + str(results["silhouette_score_euc"])
        + "\n"
    )
    f.write(
        "Silhouette score (Cosine distance): "
        + str(results["silhouette_score_cos"])
        + "\n"
    )

    f.write("\n\n")

    f.close()


def save_pickle(preds, output_path, filename):
    import pickle

    pickle.dump(preds, open(output_path + filename + ".pickle", "wb"))


def load_pickle(path):
    import pickle

    obj = pickle.load(open(path, "rb"))
    return obj


def save_i3d_model(model, model_weights, params):
    if params["model"] != "kinetics":
        print("saving i3d model...")
        output_path = params["output_path"]
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params["fold"]) + "/"
        makedirs(output_path)
        model.save(output_path + "i3d_model.h5")
        save_pickle(model_weights, output_path, "i3d_model_weights")

    else:
        print("no need to save kinetics model")


def save_ti3d_model(model, model_weights, params):
    if params["model"] != "kinetics":
        print("saving ti3d model...")
        output_path = params["output_path"]
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params["fold"]) + "/"
        output_path += str(params["model_type"]) + "/"
        output_path += str(params["iteration"]) + "/"

        makedirs(output_path)
        model.save(output_path + "ti3d_model.h5")
        save_pickle(model_weights, output_path, "ti3d_model_weights")

    else:
        print("no need to save kinetics model")


def load_i3d_model(params, NUM_CLASSES):
    if params["model"] != "kinetics":
        print("loading  i3d model...")
        output_path = params["output_path"]
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params["fold"]) + "/"
        # from keras.models import load_model
        # return load_model(output_path + 'i3d_model.h5')

        import finetune_i3d

        model = finetune_i3d.build_model(params, NUM_CLASSES)
        # model.load_weights(output_path + 'i3d_model.h5')
        model_weights = load_pickle(output_path + "i3d_model_weights.pickle")

        return model.set_weights(model_weights), model_weights


def load_ti3d_model(params):
    print("loading  i3d model...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"

    import finetune_i3d

    model = finetune_i3d.init_ti3d(params)

    # model.load_weights(output_path + 'ti3d_model.h5')
    model_weights = load_pickle(output_path + "ti3d_model_weights.pickle")

    return model.set_weights(model_weights), model_weights


def save_evm_models(evms, params, save_params=False):
    def plot_pdf_weibull(evm_class, i, scale, shape, pdf_plot_filename):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def weib(x, n, a):
            return (a / n) * (x / n) ** (a - 1) * np.exp(-((x / n) ** a))

        x = np.linspace(0.9, 1.1, 1000)
        plt.figure(0)
        plt.ylim(0, 250)
        plt.plot(x, weib(x, scale, shape))
        plt.savefig(pdf_plot_filename + str(evm_class) + "_" + str(i) + ".png")
        plt.close()

    def save_weibulls_parameters(evms, params, output_path):
        pdf_plot_filename = output_path + "weibull_plots/"
        makedirs(pdf_plot_filename)
        output_filename = "weibull_parameters.csv"

        all_parameters = []
        for evm_class, evm_model in evms.items():
            parameters_this_class = []
            for psi in evm_model.margin_weibulls:
                # scale,shape,sign,translate_amount,small_score
                psi_parameters = psi.get_params()
                parameters_this_class.append(psi_parameters)
            all_parameters.append(parameters_this_class)

        output_csv = open(output_path + output_filename, "w")

        header = "class; psi model; scale; shape; sign; translate_amount; small_score"
        output_csv.write(header + "\n")

        for evm_class, evm_models in zip(evms, all_parameters):
            for i, psi in enumerate(evm_models):
                scale = np.around(psi[0], decimals=3)
                shape = np.around(psi[1], decimals=3)
                sign = psi[2]
                translate = psi[3]
                small = np.around(psi[4], decimals=3)
                plot_pdf_weibull(
                    str(evm_class), str(i), scale, shape, pdf_plot_filename
                )

                scale = str(scale).replace(".", ",")
                shape = str(shape).replace(".", ",")
                small = str(small).replace(".", ",")

                output_csv.write(
                    str(evm_class)
                    + ";"
                    + str(i)
                    + ";"
                    + str(scale)
                    + ";"
                    + str(shape)
                    + ";"
                    + str(sign)
                    + ";"
                    + str(translate)
                    + ";"
                    + str(small)
                    + "\n"
                )

        output_csv.close()

    print("saving evms...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"
    makedirs(output_path)

    import pickle

    dbfile = open(output_path + "evms.pickle", "wb")

    pickle.dump(evms, dbfile)
    # load with pickle.load(open('evms.pickle', 'rb'))

    if save_params:
        save_weibulls_parameters(evms, params, output_path)


def load_evm_model(params):
    print("loading evms...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    # makedirs(output_path)

    import pickle

    # dbfile = open(output_path + 'evms.pickle', 'wb')

    # pickle.dump(evms, dbfile)
    # load with
    evms = pickle.load(open(output_path + "evms.pickle", "rb"))

    return evms


def save_features(
    x_train_features, x_test_features, y_train, y_test, open_y_test, params, prefix=""
):
    print("saving features...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    makedirs(output_path)

    np.save(output_path + prefix + "x_train_features.npy", x_train_features)
    np.save(output_path + prefix + "x_test_features.npy", x_test_features)
    np.save(output_path + prefix + "y_train.npy", y_train)
    np.save(output_path + prefix + "y_test.npy", y_test)
    np.save(output_path + prefix + "open_y_test.npy", open_y_test)


def save_ti3d_features(
    x_train_features, x_test_features, y_train, y_test, open_y_test, params, prefix=""
):
    print("saving features...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    makedirs(output_path)

    np.save(output_path + prefix + "x_train_features_ti3d.npy", x_train_features)
    np.save(output_path + prefix + "x_test_features_ti3d.npy", x_test_features)
    np.save(output_path + prefix + "y_train.npy", y_train)
    np.save(output_path + prefix + "y_test.npy", y_test)
    np.save(output_path + prefix + "open_y_test.npy", open_y_test)


def plot_confusion_matrix(preds, y, output_path):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sn

    font = {"size": 32}

    matplotlib.rc("font", **font)

    preds = [str(x).split("_")[2] if "_" in str(x) else str(x) for x in preds].copy()
    y = [str(x) for x in y].copy()

    # this inserts unknowns in the cm
    y[:0] = ["0"]
    preds[:0] = ["0"]

    order = list(dict.fromkeys(y))

    cm = confusion_matrix(y, preds, labels=order, normalize="true")
    fig = plt.figure(figsize=(20, 20))
    sn.heatmap(cm, annot=False, square=True, cbar_kws={"shrink": 0.8}, cmap="bwr")
    plt.yticks(rotation=0)
    plt.ylabel("True", fontsize=32, rotation=90)
    plt.xlabel("Predicted", fontsize=32, rotation=0)

    fig.savefig(output_path + "confusion_matrix.png")
    plt.close()



def plot_confusion_matrix_bars(preds, y, output_path):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    import seaborn as sn

    font = {"size": 52, "family": "arial"}

    matplotlib.rc("font", **font)

    preds = [str(x).split("_")[2] if "_" in str(x) else str(x) for x in preds].copy()
    y = [str(x) for x in y].copy()

    # this inserts unknowns in the cm

    order = list(dict.fromkeys(y))

    cm = confusion_matrix(y, preds, labels=order, normalize="true")
    diag = np.diagonal(cm) * 100.0
    print(cm, diag, cm.shape, diag.shape)

    from sklearn.linear_model import Ridge

    lr = Ridge()
    diag2 = diag.copy().reshape(-1, 1)
    x = np.array(range(101)).reshape(-1, 1)
    lr.fit(x, diag2)

    fig = plt.figure(figsize=(20, 20))
    plt.bar(range(1, 102), diag, width=0.7)
    plt.plot(
        x,
        lr.coef_ * x + lr.intercept_,
        color="red",
        linewidth=8,
        alpha=0.8,
        linestyle="solid",
    )
    # plt.yticks(rotation=0)
    plt.xticks(np.arange(1, 102, step=10))

    plt.ylabel("True Positive (%)", fontsize=52, rotation=90)
    plt.xlabel("Class", fontsize=52, rotation=0)

    fig.savefig(output_path + "confusion_matrix_bars.png")
    plt.close()



def save_predictions(preds, y, params, cm=True, prefix=""):
    print("saving predictions...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    makedirs(output_path)

    preds = np.array(preds)
    y = np.array(y)
    np.save(output_path + prefix + "preds.npy", preds)
    np.save(output_path + prefix + "phase_4_y_test.npy", y)

    if cm:
        plot_confusion_matrix(preds, y, output_path)


def save_hist(hist, params):
    if hist is not None:
        print("saving histories...")
        output_path = params["output_path"]
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params["fold"]) + "/"
        output_path += str(params["model_type"]) + "/"
        output_path += str(params["iteration"]) + "/"

        makedirs(output_path)

        import pickle

        dbfile = open(output_path + "history.pickle", "wb")

        pickle.dump(hist.history, dbfile)


def save_hist_triplet(hist, params):
    if hist is not None:
        print("saving histories...")
        output_path = params["output_path"]
        exp_id = gen_exp_id(params)

        output_path += exp_id
        output_path += str(params["fold"]) + "/"
        output_path += str(params["model_type"]) + "/"
        output_path += str(params["iteration"]) + "/"
        makedirs(output_path)

        import pickle

        dbfile = open(output_path + "history.pickle", "wb")

        pickle.dump(hist, dbfile)


def load_features(params, prefix=""):
    print("loading features")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)
    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    x = np.load(output_path + prefix + "x_train_features.npy")
    y = np.load(output_path + prefix + "x_test_features.npy")

    w = np.load(output_path + prefix + "y_train.npy")
    z = np.load(output_path + prefix + "y_test.npy")
    return x, y, w, z


def load_ti3d_features(params, prefix=""):
    print("loading ti3d features")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)
    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    x = np.load(output_path + prefix + "x_train_features_ti3d.npy")
    y = np.load(output_path + prefix + "x_test_features_ti3d.npy")

    w = np.load(output_path + prefix + "y_train.npy")
    z = np.load(output_path + prefix + "y_test.npy")
    return x, y, w, z


def read_pickle_results_file(filename="results.pickle"):
    import pickle

    infile = open(filename, "rb")
    results = pickle.load(infile)
    infile.close()
    return results


def build_observation_matrix(labels):
    labels = np.array(labels)
    len = labels.shape[0]
    obs_m = np.zeros((len, len))

    for row in range(len):
        cols = range(row, len)  # change to row+1 to avoid the diagonal
        for col in cols:
            if labels[row] == labels[col]:
                obs_m[row, col] = 1

    return obs_m


def save_full_report(forgetting, full_evaluation, params):
    print("saving full report...")
    output_path = params["output_path"]
    exp_id = gen_exp_id(params)

    output_path += exp_id
    output_path += str(params["fold"]) + "/"
    output_path += str(params["model_type"]) + "/"
    output_path += str(params["iteration"]) + "/"

    makedirs(output_path)

    output_filename = "full_report.csv"

    f = open(output_path + output_filename, "w")

    # begin forgetting-------------------------------------------------##

    n_tasks = len(forgetting)
    keys = forgetting[0].keys()

    f.write("Forgetting per task\n")

    # write the header for forgetting
    f.write("Task,")
    for k in keys:
        f.write(str(k) + ",")
    f.write("\n")

    for i, j in enumerate(forgetting):
        f.write(str(i) + ",")
        for k in keys:
            f.write(str(forgetting[i][k]) + ",")
        f.write("\n")

    # end forgetting-------------------------------------------------##

    # begin full report-------------------------------------------------##
    f.write("\n\nfull report per task\n\n")

    for i in range(n_tasks):
        f.write("\ntask " + str(i) + "\n")
        for k in keys:
            f.write(str(k) + ",")
        f.write("\n")

        for a in range(i, n_tasks):
            print("\n", full_evaluation[a][i])
            for k in keys:
                f.write(str(full_evaluation[a][i][k]) + ",")
            f.write("\n")

    print(full_evaluation)
    # end full report-------------------------------------------------##

    f.close()
