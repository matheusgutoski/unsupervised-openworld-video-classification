import numpy as np


name = "full_report.csv"
table_paths = [
    "incremental_review_results/ucf101/train_21__test_20__tail_10__cover_0.99__seed_5/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/4/",
    "incremental_review_results/ucf101/train_21__test_20__tail_10__cover_0.99__seed_6/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/4/",
    "incremental_review_results/ucf101/train_21__test_20__tail_10__cover_0.99__seed_7/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/4/",
    "incremental_review_results/ucf101/train_21__test_20__tail_10__cover_0.99__seed_8/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/4/",
    "incremental_review_results/ucf101/train_21__test_20__tail_10__cover_0.99__seed_9/0/phase_4_incremental_ti3d_incremental_evm_tail_0.2_online/4/",
]
output_file = (
    table_paths[0].split("/")[0]
    + "/"
    + table_paths[0].split("/")[1]
    + "/"
    + table_paths[0].split("/")[2]
    + "_averages.csv"
)
output = open(output_file, "w")

table_contents = []
for path in table_paths:
    table_contents.append(open(path + name, "r").readlines())


for i in range(len(table_contents[0])):
    print(i)
    cumulative = 0
    for j in range(len(table_contents)):
        if (
            "task" in table_contents[j][i]
            or "Task" in table_contents[j][i]
            or "completeness" in table_contents[j][i]
            or table_contents[j][i] == "\n"
        ):
            print("copy this")
            print(table_contents[j][i])
            output.write(table_contents[j][i])
            break
        else:
            values = np.array(table_contents[j][i].split(","))[0:-1].astype(np.float64)
            cumulative += values

    averages = cumulative / len(table_contents)
    print(averages)

    if not (
        "task" in table_contents[j][i]
        or "Task" in table_contents[j][i]
        or "completeness" in table_contents[j][i]
        or table_contents[j][i] == "\n"
    ):
        print("results")
        for a in averages:
            output.write(str(a) + ",")
        output.write("\n")
