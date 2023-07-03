import utils
import numpy as np


path = "/home/users/matheus/doutorado/openworld/openworld/prototype_results/ucf101/train_2__test_3__tail_10__cover_0.99__seed_5/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/9/"

preds = np.load(path + "preds.npy")
y = np.load(path + "phase_4_y_test.npy")
output_path = "ihardm_"
utils.plot_confusion_matrix(preds, y, output_path)
utils.plot_confusion_matrix_bars(preds, y, output_path)


methods = [
    "bic",
    "eeil",
    "ewc",
    "icarl",
    "il2m",
    "lucir",
    "lwf",
    "mas",
    "path_integral",
    "r_walk",
]

for m in methods:
    path = (
        "/home/users/matheus/doutorado/openworld/openworld/comparison_results/ucf101/train_2__test_3__tail_10__cover_0.1__seed_5/0/"
        + m
        + "/9/"
    )
    preds = np.load(path + "preds.npy")
    y = np.load(path + "phase_4_y_test.npy")
    output_path = m + "_"
    utils.plot_confusion_matrix(preds, y, output_path)
    utils.plot_confusion_matrix_bars(preds, y, output_path)
