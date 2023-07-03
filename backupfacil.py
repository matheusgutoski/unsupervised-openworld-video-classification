import numpy as np
import generator as gen
import finetune_i3d
import utils
import os
import argparse
import evm_classification as evm
import classification_metrics as metrics
import k_estimator
import hierarchical
import evaluation


import sys

sys.path.append("/home/users/matheus/doutorado/openworld/FACIL/src")

sys.path.append("/home/users/matheus/doutorado/openworld/FACIL/src/datasets")
device = "cuda"
import base_dataset as basedat
import memory_dataset as memd
from dataset_config import dataset_config
from torch.utils import data as tdata

import os
import time
import torch

torch.set_default_tensor_type("torch.FloatTensor")

import argparse
import importlib
import numpy as np
from functools import reduce

import utils_
import approach
from loggers.exp_logger import MultiLogger
from datasets.data_loader import get_loaders
from datasets.dataset_config import dataset_config
from last_layer_analysis import last_layer_analysis
from networks import tvmodels, allmodels, set_tvmodel_head_var


from keras.models import load_model
from keras import backend as K


def args_inc(argv=None, r_seed=None):
    tstart = time.time()
    # Arguments
    parser = argparse.ArgumentParser(
        description="FACIL - Framework for Analysis of Class Incremental Learning"
    )

    # miscellaneous args
    parser.add_argument("--gpu", type=int, default=1, help="GPU (default=%(default)s)")
    parser.add_argument(
        "--results-path",
        type=str,
        default="../results",
        help="Results path (default=%(default)s)",
    )
    parser.add_argument(
        "--exp-name",
        default=None,
        type=str,
        help="Experiment name (default=%(default)s)",
    )
    parser.add_argument(
        "--seed", type=int, default=r_seed, help="Random seed (default=%(default)s)"
    )
    parser.add_argument(
        "--log",
        default=["disk"],
        type=str,
        choices=["disk", "tensorboard"],
        help="Loggers used (disk, tensorboard) (default=%(default)s)",
        nargs="*",
        metavar="LOGGER",
    )
    parser.add_argument(
        "--save-models",
        action="store_true",
        help="Save trained models (default=%(default)s)",
    )
    parser.add_argument(
        "--last-layer-analysis",
        action="store_true",
        help="Plot last layer analysis (default=%(default)s)",
    )
    parser.add_argument(
        "--no-cudnn-deterministic",
        action="store_true",
        help="Disable CUDNN deterministic (default=%(default)s)",
    )
    # dataset args
    parser.add_argument(
        "--datasets",
        default=["cifar100_icarl"],
        type=str,
        choices=list(dataset_config.keys()),
        help="Dataset or datasets used (default=%(default)s)",
        nargs="+",
        metavar="DATASET",
    )
    parser.add_argument(
        "--num-workers",
        default=4,
        type=int,
        required=False,
        help="Number of subprocesses to use for dataloader (default=%(default)s)",
    )
    parser.add_argument(
        "--pin-memory",
        default=False,
        type=bool,
        required=False,
        help="Copy Tensors into CUDA pinned memory before returning them (default=%(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        required=False,
        help="Number of samples per batch to load (default=%(default)s)",
    )
    parser.add_argument(
        "--num-tasks",
        default=4,
        type=int,
        required=False,
        help="Number of tasks per dataset (default=%(default)s)",
    )
    parser.add_argument(
        "--nc-first-task",
        default=None,
        type=int,
        required=False,
        help="Number of classes of the first task (default=%(default)s)",
    )
    parser.add_argument(
        "--use-valid-only",
        action="store_true",
        help="Use validation split instead of test (default=%(default)s)",
    )
    parser.add_argument(
        "--stop-at-task",
        default=0,
        type=int,
        required=False,
        help="Stop training after specified task (default=%(default)s)",
    )
    # model args
    parser.add_argument(
        "--network",
        default="simple_mlp",
        type=str,
        choices=allmodels,
        help="Network architecture used (default=%(default)s)",
        metavar="NETWORK",
    )
    parser.add_argument(
        "--keep-existing-head",
        action="store_true",
        help="Disable removing classifier last layer (default=%(default)s)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained backbone (default=%(default)s)",
    )
    # training args
    parser.add_argument(
        "--approach",
        default="bic",
        type=str,
        choices=approach.__all__,
        help="Learning approach used (default=%(default)s)",
        metavar="APPROACH",
    )
    parser.add_argument(
        "--nepochs",
        default=30,
        type=int,
        required=False,
        help="Number of epochs per training session (default=%(default)s)",
    )
    parser.add_argument(
        "--lr",
        default=0.1,
        type=float,
        required=False,
        help="Starting learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--lr-min",
        default=1e-4,
        type=float,
        required=False,
        help="Minimum learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--lr-factor",
        default=3,
        type=float,
        required=False,
        help="Learning rate decreasing factor (default=%(default)s)",
    )
    parser.add_argument(
        "--lr-patience",
        default=5,
        type=int,
        required=False,
        help="Maximum patience to wait before decreasing learning rate (default=%(default)s)",
    )
    parser.add_argument(
        "--clipping",
        default=10000,
        type=float,
        required=False,
        help="Clip gradient norm (default=%(default)s)",
    )
    parser.add_argument(
        "--momentum",
        default=0.0,
        type=float,
        required=False,
        help="Momentum factor (default=%(default)s)",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.0,
        type=float,
        required=False,
        help="Weight decay (L2 penalty) (default=%(default)s)",
    )
    parser.add_argument(
        "--warmup-nepochs",
        default=0,
        type=int,
        required=False,
        help="Number of warm-up epochs (default=%(default)s)",
    )
    parser.add_argument(
        "--warmup-lr-factor",
        default=1.0,
        type=float,
        required=False,
        help="Warm-up learning rate factor (default=%(default)s)",
    )
    parser.add_argument(
        "--multi-softmax",
        action="store_true",
        help="Apply separate softmax for each task (default=%(default)s)",
    )
    parser.add_argument(
        "--fix-bn",
        action="store_true",
        help="Fix batch normalization after first task (default=%(default)s)",
    )
    parser.add_argument(
        "--eval-on-train",
        action="store_true",
        help="Show train loss and accuracy (default=%(default)s)",
    )
    # gridsearch args
    parser.add_argument(
        "--gridsearch-tasks",
        default=-1,
        type=int,
        help="Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)",
    )

    # Args -- Incremental Learning Framework
    args, extra_args = parser.parse_known_args(argv)
    args.results_path = os.path.expanduser(args.results_path)
    base_kwargs = dict(
        nepochs=args.nepochs,
        lr=args.lr,
        lr_min=args.lr_min,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        clipgrad=args.clipping,
        momentum=args.momentum,
        wd=args.weight_decay,
        multi_softmax=args.multi_softmax,
        wu_nepochs=args.warmup_nepochs,
        wu_lr_factor=args.warmup_lr_factor,
        fix_bn=args.fix_bn,
        eval_on_train=args.eval_on_train,
    )

    if args.no_cudnn_deterministic:
        print("WARNING: CUDNN Deterministic will be disabled.")
        utils_.cudnn_deterministic = False

    utils_.seed_everything(seed=args.seed)
    print("=" * 108)
    print("Arguments =")
    for arg in np.sort(list(vars(args).keys())):
        print("\t" + arg + ":", getattr(args, arg))
    print("=" * 108)

    # Args -- CUDA
    if torch.cuda.is_available():
        # torch.cuda.set_device(args.gpu)
        device = "cuda"
    else:
        print("WARNING: [CUDA unavailable] Using CPU instead!")
        device = "cpu"
    # Multiple gpus
    # if torch.cuda.device_count() > 1:
    #     self.C = torch.nn.DataParallel(C)
    #     self.C.to(self.device)
    ####################################################################################################################

    # Args -- Network
    from networks.network import LLL_Net

    if args.network in tvmodels:  # torchvision models
        tvnet = getattr(
            importlib.import_module(name="torchvision.models"), args.network
        )
        if args.network == "googlenet":
            init_model = tvnet(pretrained=args.pretrained, aux_logits=False)
        else:
            init_model = tvnet(pretrained=args.pretrained)
        set_tvmodel_head_var(init_model)
    else:  # other models declared in networks package's init
        net = getattr(importlib.import_module(name="networks"), args.network)
        # WARNING: fixed to pretrained False for other model (non-torchvision)
        init_model = net(pretrained=False)

    # Args -- Continual Learning Approach
    from approach.incremental_learning import Inc_Learning_Appr

    Appr = getattr(importlib.import_module(name="approach." + args.approach), "Appr")
    assert issubclass(Appr, Inc_Learning_Appr)
    appr_args, extra_args = Appr.extra_parser(extra_args)
    print("Approach arguments =")
    for arg in np.sort(list(vars(appr_args).keys())):
        print("\t" + arg + ":", getattr(appr_args, arg))
    print("=" * 108)

    # Args -- Exemplars Management
    from datasets.exemplars_dataset import ExemplarsDataset

    Appr_ExemplarsDataset = Appr.exemplars_dataset_class()
    if Appr_ExemplarsDataset:
        assert issubclass(Appr_ExemplarsDataset, ExemplarsDataset)
        appr_exemplars_dataset_args, extra_args = Appr_ExemplarsDataset.extra_parser(
            extra_args
        )
        print("Exemplars dataset arguments =")
        for arg in np.sort(list(vars(appr_exemplars_dataset_args).keys())):
            print("\t" + arg + ":", getattr(appr_exemplars_dataset_args, arg))
        print("=" * 108)
    else:
        appr_exemplars_dataset_args = argparse.Namespace()

    # Args -- GridSearch
    if args.gridsearch_tasks > 0:
        from gridsearch import GridSearch

        gs_args, extra_args = GridSearch.extra_parser(extra_args)
        Appr_finetuning = getattr(
            importlib.import_module(name="approach.finetuning"), "Appr"
        )
        assert issubclass(Appr_finetuning, Inc_Learning_Appr)
        GridSearch_ExemplarsDataset = Appr.exemplars_dataset_class()
        print("GridSearch arguments =")
        for arg in np.sort(list(vars(gs_args).keys())):
            print("\t" + arg + ":", getattr(gs_args, arg))
        print("=" * 108)

    assert len(extra_args) == 0, "Unused args: {}".format(" ".join(extra_args))
    ####################################################################################################################

    # Log all arguments
    full_exp_name = (
        reduce((lambda x, y: x[0] + y[0]), args.datasets)
        if len(args.datasets) > 0
        else args.datasets[0]
    )
    full_exp_name += "_" + args.approach
    if args.exp_name is not None:
        full_exp_name += "_" + args.exp_name
    logger = MultiLogger(
        args.results_path, full_exp_name, loggers=args.log, save_models=args.save_models
    )
    logger.log_args(
        argparse.Namespace(
            **args.__dict__,
            **appr_args.__dict__,
            **appr_exemplars_dataset_args.__dict__
        )
    )

    print("dataset loader")
    # Loaders
    utils_.seed_everything(seed=args.seed)
    trn_loader, val_loader, tst_loader, taskcla = get_loaders(
        args.datasets,
        args.num_tasks,
        args.nc_first_task,
        args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    # Apply arguments for loaders
    if args.use_valid_only:
        tst_loader = val_loader
    max_task = len(taskcla) if args.stop_at_task == 0 else args.stop_at_task

    # Network and Approach instances
    utils_.seed_everything(seed=args.seed)
    net = LLL_Net(init_model, remove_existing_head=not args.keep_existing_head)
    utils_.seed_everything(seed=args.seed)
    # taking transformations and class indices from first train dataset
    first_train_ds = trn_loader[0].dataset
    transform, class_indices = first_train_ds.transform, first_train_ds.class_indices
    appr_kwargs = {**base_kwargs, **dict(logger=logger, **appr_args.__dict__)}
    if Appr_ExemplarsDataset:
        appr_kwargs["exemplars_dataset"] = Appr_ExemplarsDataset(
            transform, class_indices, **appr_exemplars_dataset_args.__dict__
        )
    utils_.seed_everything(seed=args.seed)
    net.to(device)
    appr = Appr(net, device, **appr_kwargs)

    return appr, trn_loader, val_loader, tst_loader, taskcla, net, device

    """
    # GridSearch
    if args.gridsearch_tasks > 0:
        ft_kwargs = {**base_kwargs, **dict(logger=logger,
                                           exemplars_dataset=GridSearch_ExemplarsDataset(transform, class_indices))}
        appr_ft = Appr_finetuning(net, device, **ft_kwargs)
        gridsearch = GridSearch(appr_ft, args.seed, gs_args.gridsearch_config, gs_args.gridsearch_acc_drop_thr,
                                gs_args.gridsearch_hparam_decay, gs_args.gridsearch_max_num_searches)

    # Loop tasks
    print(taskcla)
    acc_taw = np.zeros((max_task, max_task))
    acc_tag = np.zeros((max_task, max_task))
    forg_taw = np.zeros((max_task, max_task))
    forg_tag = np.zeros((max_task, max_task))
    for t, (_, ncla) in enumerate(taskcla):
        # Early stop tasks if flag
        if t >= max_task:
            continue
        print('*' * 108)
        print('Task {:2d}'.format(t))
        print('*' * 108)

        # Add head for current task
        net.add_head(taskcla[t][1])
        net.to(device)

        # GridSearch
        if t < args.gridsearch_tasks:

            # Search for best finetuning learning rate -- Maximal Plasticity Search
            print('LR GridSearch')
            best_ft_acc, best_ft_lr = gridsearch.search_lr(appr.model, t, trn_loader[t], val_loader[t])
            # Apply to approach
            appr.lr = best_ft_lr
            gen_params = gridsearch.gs_config.get_params('general')
            for k, v in gen_params.items():
                if not isinstance(v, list):
                    setattr(appr, k, v)

            # Search for best forgetting/intransigence tradeoff -- Stability Decay
            print('Trade-off GridSearch')
            best_tradeoff, tradeoff_name = gridsearch.search_tradeoff(args.approach, appr,
                                                                      t, trn_loader[t], val_loader[t], best_ft_acc)
            # Apply to approach
            if tradeoff_name is not None:
                setattr(appr, tradeoff_name, best_tradeoff)

            print('-' * 108)

        # Train
        appr.train(t, trn_loader[t], val_loader[t])
        print('-' * 108)

        # Test
        for u in range(t + 1):
            test_loss, acc_taw[t, u], acc_tag[t, u] = appr.eval(u, tst_loader[u])
            if u < t:
                forg_taw[t, u] = acc_taw[:t, u].max(0) - acc_taw[t, u]
                forg_tag[t, u] = acc_tag[:t, u].max(0) - acc_tag[t, u]
            print('>>> Test on task {:2d} : loss={:.3f} | TAw acc={:5.1f}%, forg={:5.1f}%'
                  '| TAg acc={:5.1f}%, forg={:5.1f}% <<<'.format(u, test_loss,
                                                                 100 * acc_taw[t, u], 100 * forg_taw[t, u],
                                                                 100 * acc_tag[t, u], 100 * forg_tag[t, u]))
            logger.log_scalar(task=t, iter=u, name='loss', group='test', value=test_loss)
            logger.log_scalar(task=t, iter=u, name='acc_taw', group='test', value=100 * acc_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='acc_tag', group='test', value=100 * acc_tag[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_taw', group='test', value=100 * forg_taw[t, u])
            logger.log_scalar(task=t, iter=u, name='forg_tag', group='test', value=100 * forg_tag[t, u])

        # Save
        print('Save at ' + os.path.join(args.results_path, full_exp_name))
        logger.log_result(acc_taw, name="acc_taw", step=t)
        logger.log_result(acc_tag, name="acc_tag", step=t)
        logger.log_result(forg_taw, name="forg_taw", step=t)
        logger.log_result(forg_tag, name="forg_tag", step=t)
        logger.save_model(net.state_dict(), task=t)
        logger.log_result(acc_taw.sum(1) / np.tril(np.ones(acc_taw.shape[0])).sum(1), name="avg_accs_taw", step=t)
        logger.log_result(acc_tag.sum(1) / np.tril(np.ones(acc_tag.shape[0])).sum(1), name="avg_accs_tag", step=t)
        aux = np.tril(np.repeat([[tdata[1] for tdata in taskcla[:max_task]]], max_task, axis=0))
        logger.log_result((acc_taw * aux).sum(1) / aux.sum(1), name="wavg_accs_taw", step=t)
        logger.log_result((acc_tag * aux).sum(1) / aux.sum(1), name="wavg_accs_tag", step=t)

        # Last layer analysis
        if args.last_layer_analysis:
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)

            # Output sorted weights and biases
            weights, biases = last_layer_analysis(net.heads, t, taskcla, y_lim=True, sort_weights=True)
            logger.log_figure(name='weights', iter=t, figure=weights)
            logger.log_figure(name='bias', iter=t, figure=biases)
    # Print Summary
    utils_.print_summary(acc_taw, acc_tag, forg_taw, forg_tag)
    print('[Elapsed time = {:.1f} h]'.format((time.time() - tstart) / (60 * 60)))
    print('Done!')

    return acc_taw, acc_tag, forg_taw, forg_tag, logger.exp_path
    ####################################################################################################################
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Use which model?",
        type=str,
        choices=["kinetics", "ucf101"],
        default="ucf101",
    )
    parser.add_argument("--list_id", help="Video list id?", type=int, default=0)
    parser.add_argument(
        "--n_train_classes",
        help="How many classes at training time?",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--n_test_classes", help="How many classes at test time?", type=int, default=3
    )
    parser.add_argument("--n_folds", help="How many folds?", type=int, default=30)
    parser.add_argument(
        "--train_ratio",
        help="Fraction of the training set to use for training the cnn. The rest is used for validation",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--num_frames",
        help="Number of frames in the temporal dimension",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--num_frames_test",
        help="Number of frames in the temporal dimension during test time",
        type=int,
        default=250,
    )
    parser.add_argument("--frame_height", help="Frame height", type=int, default=224)
    parser.add_argument("--frame_width", help="Frame width", type=int, default=224)
    parser.add_argument(
        "--max_epochs", help="maximum number of epochs", type=int, default=10
    )
    parser.add_argument(
        "--triplet_epochs",
        help="triplet net max number of epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--triplet_lr", help="triplet learning rate with sgd", type=float, default=0.001
    )
    parser.add_argument(
        "--margin", help="triplet net margin parameter", type=float, default=0.2
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=6)
    parser.add_argument(
        "--triplet_batch_size", help="batch size of triplet net", type=int, default=128
    )
    parser.add_argument(
        "--tail_size",
        help="weibull tail size for evm. If < 1, uses ratio of data instead",
        type=float,
        default=10,
    )
    parser.add_argument(
        "--cover_threshold", help="evm cover threshold", type=float, default=0.1
    )
    parser.add_argument(
        "--classification_threshold",
        help="probability threshold for accepting points in evm",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--output_path",
        help="Where to output results",
        type=str,
        default="comparison_results/",
    )
    parser.add_argument(
        "--ti3d_type",
        help="ti3d type",
        type=str,
        default="incremental",
        choices=["incremental", "gold", "fixed"],
    )
    parser.add_argument("--online", help="train ti3d in 1 epoch", action="store_true")
    parser.add_argument(
        "--incremental_evm",
        help="use incremental evm instead of gold standard",
        action="store_true",
    )
    parser.add_argument(
        "--seed", help="Random seed. 123450 sets seed to random", type=int, default=5
    )
    parser.add_argument("--gpu", help="gpu id", type=int)
    parser.add_argument("--expid", help="exp id", type=str)
    parser.add_argument(
        "--incremental_model", help="which incremental model to use", type=str
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # set all params
    params = {}
    params["min_classes"] = 10
    params["max_classes"] = 11
    params["seed"] = args.seed
    params["init_seed"] = params["seed"]
    params["model_type"] = "cnn"

    np.random.seed(params["seed"])

    params["model"] = args.model
    params["list_id"] = args.list_id
    params["n_train_classes"] = args.n_train_classes
    params["n_test_classes"] = args.n_test_classes
    params["n_folds"] = args.n_folds
    params["output_path"] = args.output_path
    params["fold"] = 0

    # ignore errors - ignores classification errors until phase 4. Applies to EVM rejection and clustering
    params["ignore_errors"] = True

    # parameters for the cnn
    params["train_ratio"] = args.train_ratio
    params["num_frames"] = args.num_frames
    params["num_frames_test"] = args.num_frames_test
    params["frame_height"] = args.frame_height
    params["frame_width"] = args.frame_width
    params["max_epochs"] = args.max_epochs
    params["batch_size"] = args.batch_size

    # parameters for the triplet net
    params["triplet_epochs"] = args.triplet_epochs
    params["triplet_epochs_incremental"] = 10
    params["margin"] = args.margin
    params["triplet_lr"] = args.triplet_lr
    params["triplet_batch_size"] = args.triplet_batch_size

    # parameters for the evm
    params["tail_size"] = args.tail_size
    params["cover_threshold"] = args.cover_threshold
    params["classification_threshold"] = args.classification_threshold
    # multi_classification_threshold = [0.00000001,0.0000001,0.000001,0.00001,0.0001,0.001,0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99,0.999, 0.9999]
    # multi_classification_threshold = [0.001]

    params["iteration"] = 0
    params["estimate_new_data_evs"] = False

    # params['online'] = False
    # params['online'] = True
    params["online"] = args.online
    params["incremental_evm"] = args.incremental_evm
    params["ti3d_type"] = args.ti3d_type

    params["incremental_model"] = args.incremental_model
    # get filenames
    filenames = gen.get_filenames()
    all_categories = gen.get_all_categories(filenames)
    dict_map = utils.map_labels(all_categories)

    # get class list
    unique_classes = np.unique([x.split("/")[0] for x in filenames]).tolist()

    PATH_TO_FRAMES = "/home/users/datasets/UCF-101_opticalflow/"

    train_i3d_features = []
    train_i3d_labels = []

    test_i3d_features = []
    test_i3d_labels = []

    all_results = []

    # PHASE 1									------------------------

    # perform train/test split

    train, train_labels, test, test_labels = utils.train_test_split_groups(
        filenames, unique_classes, params
    )
    print(len(train_labels), len(test_labels), len(train_labels) + len(test_labels))
    int_train_labels = utils.convert_labels_to_int(train_labels, dict_map)
    int_test_labels = utils.convert_labels_to_int(test_labels, dict_map)

    # select initial classes

    perm = np.random.permutation(np.array(unique_classes).shape[0])
    class_shuffle = np.array(unique_classes)[perm]
    print(unique_classes, class_shuffle)

    # pick n classes between minclasses and maxclasses
    # initial_n_classes = np.random.randint(params['min_classes'],params['max_classes'])
    initial_n_classes = 11
    initial_classes = class_shuffle[0:initial_n_classes]
    current_classes = initial_classes.copy()
    remaining_classes = class_shuffle[initial_n_classes:]

    initial_train = []
    initial_train_labels = []
    initial_test = []
    initial_test_labels = []

    for t, tl in zip(train, train_labels):
        if tl in initial_classes:
            initial_train.append(t)
            initial_train_labels.append(tl)

    for t, tl in zip(test, test_labels):
        if tl in initial_classes:
            initial_test.append(t)
            initial_test_labels.append(tl)

    int_initial_train_labels = utils.convert_labels_to_int(
        initial_train_labels, dict_map
    )
    int_initial_test_labels = utils.convert_labels_to_int(initial_test_labels, dict_map)

    int_initial_classes = utils.convert_labels_to_int(initial_classes, dict_map)

    open_y_test = [
        x if x in int_initial_train_labels else 0 for x in int_initial_test_labels
    ]
    class_history = []
    int_class_history = []
    class_history.append(initial_classes)
    # int_class_history.append(int_initial_classes)
    total_classes = initial_n_classes
    print(np.unique(int_initial_train_labels), np.unique(int_initial_test_labels))

    # train i3d
    all_categories_train_fold = gen.get_all_categories(initial_train)
    dict_map_train_fold = utils.map_labels(all_categories_train_fold)
    all_categories_test_fold = gen.get_all_categories(initial_test)
    dict_map_test_fold = utils.map_labels(all_categories_test_fold)

    all_categories_test_fold_full = gen.get_all_categories(test)
    dict_map_test_fold_full = utils.map_labels(all_categories_test_fold_full)

    try:
        params["model_type"] = "cnn"

        # model, model_weights = utils.load_i3d_model(params,initial_n_classes)
        (
            train_features,
            test_features,
            int_train_labels,
            int_test_labels,
        ) = utils.load_features(params)
    except Exception as e:
        params["model_type"] = "cnn"
        model, hist_cnn, model_weights = finetune_i3d.finetune(
            initial_train, int_initial_train_labels, dict_map_train_fold, params
        )
        utils.save_i3d_model(model, model_weights, params)  # very expensive storage
        train_features, test_features = finetune_i3d.extract_features(
            model_weights,
            train,
            initial_train_labels,
            test,
            test_labels,
            dict_map_test_fold_full,
            params,
        )
        utils.save_features(
            train_features,
            test_features,
            int_train_labels,
            int_test_labels,
            int_test_labels,
            params,
        )
    print(train_features.shape, test_features.shape)

    print(np.unique(int_train_labels), np.unique(int_test_labels))
    task_classes = {}
    n_tasks = 10
    initial_n_classes = 11
    n_classes_per_task = 10
    total_classes = 101

    int_class_shuffle = utils.convert_labels_to_int(class_shuffle, dict_map)

    seq_dict_map = {}
    for i in range(101):
        seq_dict_map[int_class_shuffle[i]] = i

    print(seq_dict_map)
    c = 0
    for n in range(n_tasks):
        if n == 0:
            task_classes[n] = utils.convert_labels_to_int(
                class_shuffle[0:initial_n_classes], dict_map
            )
            c += initial_n_classes
        else:
            task_classes[n] = utils.convert_labels_to_int(
                class_shuffle[c : c + n_classes_per_task], dict_map
            )
            c += n_classes_per_task

    for k, v in task_classes.items():
        print(k, v)

        """
		new_train = []
		new_train_labels = []
		new_test = []
		new_test_labels = []

		for t, tl in zip(train, train_labels):
			if tl in new_classes:
				new_train.append(t)
				new_train_labels.append(tl)

		for t, tl in zip(test, test_labels):
			if tl in new_classes:
				new_test.append(t)
				new_test_labels.append(tl)

		int_new_train_labels = utils.convert_labels_to_int(new_train_labels, dict_map)
		int_new_test_labels = utils.convert_labels_to_int(new_test_labels, dict_map)
		"""

    data = {}

    for tt in range(n_tasks):
        data[tt] = {}
        data[tt]["name"] = "task-" + str(tt)
        data[tt]["trn"] = {"x": [], "y": []}
        data[tt]["val"] = {"x": [], "y": []}
        data[tt]["tst"] = {"x": [], "y": []}

    for this_task in range(n_tasks):
        new_train = []
        new_train_labels = []
        new_test = []
        new_test_labels = []

        for t, tl in zip(train_features, int_train_labels):
            if tl in task_classes[this_task]:
                new_train.append(t)
                new_train_labels.append(tl)

        for t, tl in zip(test_features, int_test_labels):
            if tl in task_classes[this_task]:
                new_test.append(t)
                new_test_labels.append(tl)

        print(
            np.unique(new_train_labels),
            np.unique(new_test_labels),
            np.unique(task_classes[this_task]),
        )
        print(task_classes[this_task])

        # for l in new_train_labels:
        # print(l, seq_dict_map[l])
        # print([seq_dict_map[x] for x in new_train_labels])

        for t, tl in zip(new_train, new_train_labels):
            data[this_task]["trn"]["x"].append(t)
            data[this_task]["trn"]["y"].append(seq_dict_map[tl])

        for t, tl in zip(new_test, new_test_labels):
            data[this_task]["tst"]["x"].append(t)
            data[this_task]["tst"]["y"].append(seq_dict_map[tl])

            data[this_task]["val"]["x"].append(t)  # fix later
            data[this_task]["val"]["y"].append(seq_dict_map[tl])

        """
		#dummy_data = np.zeros((1024))
		for a in range(200):
			if this_task == 0:
				low = 0
				high = 11 # up to but no including
			else:
				low = 10*this_task + 1
				high = low + 10
			dummy_label = np.random.randint(low, high)

			data[this_task]['trn']['x'].append(dummy_data)
			data[this_task]['trn']['y'].append(dummy_label)

			data[this_task]['tst']['x'].append(dummy_data)
			data[this_task]['tst']['y'].append(dummy_label)

			data[this_task]['val']['x'].append(dummy_data)
			data[this_task]['val']['y'].append(dummy_label)
		"""

    for tt in range(n_tasks):
        data[tt]["ncla"] = len(np.unique(data[tt]["trn"]["y"]))

    # convert them to numpy arrays
    for tt in data.keys():
        for split in ["trn", "val", "tst"]:
            data[tt][split]["x"] = np.asarray(data[tt][split]["x"])
            print(data[tt][split]["x"].shape, tt)
    # other
    n = 0
    taskcla = []
    for t in data.keys():
        taskcla.append((t, data[t]["ncla"]))
        n += data[t]["ncla"]
    data["ncla"] = n

    Dataset = memd.MemoryDataset

    trn_dset, val_dset, tst_dset = [], [], []
    all_data = data.copy()

    trn_transform, tst_transform = [], []
    offset = 0
    for task in range(n_tasks):
        trn_dset.append(Dataset(all_data[task]["trn"], transform=None))
        val_dset.append(Dataset(all_data[task]["val"], transform=None))
        tst_dset.append(Dataset(all_data[task]["tst"], transform=None))
        offset += taskcla[task][1]

    # loaders
    trn_load, val_load, tst_load = [], [], []
    batch_size = 16
    num_workers = 8

    for tt in range(n_tasks):
        trn_load.append(
            tdata.DataLoader(
                trn_dset[tt],
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        )
        val_load.append(
            tdata.DataLoader(
                val_dset[tt],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        )
        tst_load.append(
            tdata.DataLoader(
                tst_dset[tt],
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        )

    for a, b in trn_load[0]:
        print(a.shape, b.shape)


    train_i3d_features.append(train_features)
    train_i3d_labels.append(int_initial_train_labels)
    test_i3d_features.append(test_features)
    test_i3d_labels.append(int_initial_test_labels)

    params["openness"] = 0

    params["model_type"] = params["incremental_model"]
    # classify i3d model data with incremental classifier and get classification metrics

    appr, ___, _____, _________, ____________________, net, device = args_inc(
        r_seed=params["seed"]
    )

    import torch

    for t, (_, ncla) in enumerate(taskcla):
        int_class_history.append(np.unique(data[t]["tst"]["y"]))
        all_preds = []
        all_targets = []

        print(device)
        net.to(device)
        net.add_head(taskcla[t][1])

        appr.train(t, trn_load[t], val_load[t])

        for ii in range(t + 1):
            print("aa", ii)
            aa, bb, cc, preds, targ = appr.eval(ii, tst_load[ii])
            for p, t in zip(preds, targ):
                outputs = [
                    torch.nn.functional.log_softmax(output, dim=1) for output in p
                ]
                pred = torch.cat(outputs, dim=1).argmax(1)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(t.cpu().numpy())
        all_preds = np.concatenate((all_preds))
        all_targets = np.concatenate((all_targets))

        evaluation.single_evaluation_clustering(
            test_features, all_targets, all_preds, params
        )

        dict = {}
        dict["x"] = test_features.copy()
        dict["y"] = all_targets.copy()
        dict["preds"] = all_preds.copy()
        dict["tasks"] = int_class_history.copy()
        all_results.append(dict.copy())

        forgetting, full_evaluation = evaluation.full_evaluation(all_results, params)

        utils.save_full_report(forgetting, full_evaluation, params)
        utils.save_predictions(all_preds, all_targets, params, cm=True)
        params["iteration"] += 1

    sys.exit(0)

    # pred = clf.predict(evms_triplet, test_features, params)
    pred = np.zeros((test_features.shape[0]))
    print(len(open_y_test), len(pred))
    evaluation.single_evaluation_openset(open_y_test, pred, params)
    evaluation.single_evaluation_clustering(test_features, open_y_test, pred, params)

    dict = {}
    dict["x"] = test_features.copy()
    dict["y"] = open_y_test.copy()
    dict["preds"] = pred.copy()
    dict["tasks"] = [int_initial_classes.copy()]
    all_results.append(dict.copy())

    # end phase 1


    # phase 2					 				-----------------------

    while total_classes < 101:
        params["iteration"] += 1
        # select z new classes

        new_n_classes = np.random.randint(params["min_classes"], params["max_classes"])
        new_classes = class_shuffle[total_classes : total_classes + new_n_classes]
        total_classes += new_n_classes
        print("new selected classes:", new_classes)
        # new data

        new_train = []
        new_train_labels = []
        new_test = []
        new_test_labels = []

        for t, tl in zip(train, train_labels):
            if tl in new_classes:
                new_train.append(t)
                new_train_labels.append(tl)

        for t, tl in zip(test, test_labels):
            if tl in new_classes:
                new_test.append(t)
                new_test_labels.append(tl)

        int_new_train_labels = utils.convert_labels_to_int(new_train_labels, dict_map)
        int_new_test_labels = utils.convert_labels_to_int(new_test_labels, dict_map)

        # extract features

        all_categories_new_train = gen.get_all_categories(new_train)
        all_categories_new_test = gen.get_all_categories(new_test)

        dict_map_new_train = utils.map_labels(all_categories_new_train)
        dict_map_new_test = utils.map_labels(all_categories_new_test)

        try:
            params["model_type"] = "cnn"
            (
                new_train_features,
                new_test_features,
                int_new_train_labels,
                int_new_test_labels,
            ) = utils.load_features(params, prefix="phase_2")
            # print(np.unique(int_new_test_labels))

        except Exception as e:
            print(e)
            params["model_type"] = "cnn"
            new_train_features = finetune_i3d.extract_features_single(
                model_weights,
                new_train,
                new_train_labels,
                dict_map_new_train,
                params,
                len(np.unique(initial_train_labels)),
            )
            new_test_features = finetune_i3d.extract_features_single(
                model_weights,
                new_test,
                new_test_labels,
                dict_map_new_test,
                params,
                len(np.unique(initial_train_labels)),
            )
            utils.save_features(
                new_train_features,
                new_test_features,
                int_new_train_labels,
                int_new_test_labels,
                int_new_test_labels,
                params,
                prefix="phase_2",
            )

        train_i3d_features.append(new_train_features)
        test_i3d_features.append(new_test_features)
        train_i3d_labels.append(int_new_train_labels)
        test_i3d_labels.append(int_new_test_labels)

        # get rejected set (train)
        if params["ignore_errors"] == True:
            rejected_set_features_i3d = np.array(new_train_features)
            rejected_set_labels = np.array(int_new_train_labels)
        else:
            raise Exception

        # get all known classes
        known_classes = np.concatenate(class_history).ravel()
        int_known_classes = utils.convert_labels_to_int(known_classes, dict_map)

        flattened_test_i3d_features, flattened_test_i3d_labels = np.concatenate(
            test_i3d_features
        ), np.concatenate(test_i3d_labels)
        flattened_train_i3d_features, flattened_train_i3d_labels = np.concatenate(
            train_i3d_features
        ), np.concatenate(train_i3d_labels)

        full_test_labels = flattened_test_i3d_labels.copy()
        full_open_test_labels = [
            x if x in int_known_classes else 0 for x in full_test_labels
        ]

        class_history.append(new_classes)
        int_class_history.append(utils.convert_labels_to_int(new_classes, dict_map))

        print(int_class_history)
        print(np.unique(full_test_labels))
        print("int class his:", int_class_history)
        print("full_open_test_labels", full_open_test_labels)
        print("full test labels", full_test_labels)


        # end phase 2

        # phase 3								-----------------------

        # estimate number of clusters in the rejected set

        top_k = 5
        estimated_k, gaps = k_estimator.estimate_dendrogap(
            rejected_set_features_i3d, top_k, normalize_data=True
        )
        estimated_k = k_estimator.best_silhouette(
            rejected_set_features_i3d, estimated_k, metric="cosine"
        )
        print("estimated k:", estimated_k)
        print("true k", len(np.unique(rejected_set_labels)))

        # cluster with hierarchical agglomerative ward clustering

        # perform hierarchical
        print("Performing hierarchical clustering with ward linkage")
        # get rejected set labels

        if params["ignore_errors"] == True:
            hierarchical_preds = rejected_set_labels
        else:
            hierarchical_preds = hierarchical.hierarchical(
                rejected_set_features_i3d,
                n_clusters=estimated_k,
                affinity="euclidean",
                linkage="ward",
                distance_threshold=None,
                normalize_data=True,
            )

        # assign labels
        hierarchical_preds = [
            "new_class_" + str(x) + "_iter_" + str(params["iteration"])
            for x in hierarchical_preds
        ]

        # evaluate clustering performance
        params["model_type"] = "phase_3"
        evaluation.single_evaluation_clustering(
            rejected_set_features_i3d, rejected_set_labels, hierarchical_preds, params
        )

        # end phase 3

        # phase 4								------------------------

        params["model_type"] = (
            "phase_4_incremental_ti3d_incremental_evm_tail_"
            + str(params["tail_size"])
            + "_online"
        )

        print("phase 4 clf predict")
        # clf.train(flattened_train_i3d_features, flattened_train_i3d_labels)
        # preds = clf.predict(flattened_test_i3d_features, params)
        preds = np.zeros((flattened_test_i3d_features.shape[0]))

        evaluation.single_evaluation_clustering(
            flattened_test_i3d_features, full_test_labels, preds, params
        )

        dict = {}
        dict["x"] = flattened_train_i3d_features.copy()
        dict["y"] = full_test_labels.copy()
        dict["preds"] = preds.copy()
        dict["tasks"] = int_class_history.copy()
        all_results.append(dict.copy())

        forgetting, full_evaluation = evaluation.full_evaluation(all_results, params)

        utils.save_full_report(forgetting, full_evaluation, params)
        utils.save_predictions(preds, full_test_labels, params, cm=True)

