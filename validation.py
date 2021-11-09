import os
import pickle
from functools import partial

from ray.rllib.examples.centralized_critic import nn
from ray.rllib.train import torch
from ray.tune import CLIReporter
from ray.tune.examples.cifar10_pytorch import Net
from ray.tune.schedulers import ASHAScheduler
from ray.tune.tests.ext_pytorch import test_accuracy
from ray.util.sgd.v2.trainer import tune
from torch import optim
from torch.nn import CrossEntropyLoss

from dataset import SWaTDataset
from model import ATTAINModel
from settings import Config
from train import train


def cross_validation(data, candidates):
    """

    :param data: swat/wadi/batadal
    :param candidates: all possible hyperparameters
    :return: (best_f1,best_parameter)
    """
    length = len(data)
    valid_size = length / 10
    best_f1, best_params = 0.0, None
    for candidate in zip(candidates):
        config.batch_size = candidate.batch_size
        config.hidden_size = candidate.hidden_size
        model = ATTAINModel(config)
        criterion = CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        f1_sum = 0.0
        for i in range(10):
            valid_start = i * valid_size
            valid_end = i * (valid_size) - 1
            valid_set = [data[j] for j in range(len(data)) if valid_start <= j <= valid_end]
            train_set = [data[j] for j in range(len(data)) if valid_start <= j <= valid_end]
            f1 = train(model, train_set, valid_set, criterion, optimizer)
            f1_sum += f1
        f1_avg = f1_sum / 10
        if f1_avg > best_f1:
            best_f1 = f1_avg
            best_params = candidate

    return best_f1, best_params


def ray_tune(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    config = Config()
    dataset_settings = {
        "swat": {
            "train_dataset": SWaTDataset(config, training=True),
            "test_dataset": SWaTDataset(config, training=False),
            "attack_log_path": config.swat_attack_log_path
        },
        # "wadi": {
        #     "train_dataset": WADIDataset(config, training=True),
        #     "test_dataset": WADIDataset(config, training=False)
        # "attack_log_path": config.wadi_attack_log_path
        # },
        # "batadal": {
        #     "train_dataset": BATADALDataset(config, training=True),
        #     "test_dataset": BATADALDataset(config, train(False))
        # "attack_log_path": config.batadal_attack_log_path
        # }
        # "phm": {
        #     "train_dataset": PHMDataset(config, training=True),
        #     "test_dataset": PHMDataset(config, training=False),
        #     "attack_log_path": config.phm_attack_log_path
        # },
        # "gas": {
        #     "train_dataset": GASDataset(config, training=True),
        #     "test_dataset": GASDataset(config, training=False),
        #     "attack_log_path": config.gas_attack_log_path
        # },
    }
    train_dataset = dataset_settings["swat"]["train_dataset"]
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))

    best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == '__main__':
    config = Config()
    best_f1, best_params = 0.0, None
    data_pkls = [config.swat_valid_pkl_path, config.wadi_valid_pkl_path, config.batadal_valid_pkl_path]
    hyperparameters = {
        "batch_size": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "hidden_size": [16, 32, 64, 128, 256, 512, 10, 20, 40, 80, 100]
    }

    for data_pkl in data_pkls:
        data = pickle.load(open(data_pkl, "rb"))
        f1, params = cross_validation(data, hyperparameters)
