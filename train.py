import os

import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import numpy as np
# train loop
from torch.utils.data import DataLoader

from cl import CurriculumLearning
from dataset import SWaTDataset, WADIDataset, BATADALDataset, PHMDataset, GASDataset
from model import LSTMCNNModel, CUSUMModel, GANModel, ATTAINModel
from settings import Config
import torch.optim as optim
from atpbar import atpbar

from utils import LabelMaker


def train(model, train_loader, evaluate_loader, criterion, optimizer, label_maker, cl, shuffled_train_loader=None):
    """main training loop"""
    # train_iter = tqdm(train_loader)

    best_precision, best_recall, best_f1 = 0.0, 0.0, 0.0
    cl_real_vec = []
    cl_dtm_preds = []
    cl_labels = []
    for epoch_i in range(config.n_epochs):
        if shuffled_train_loader is not None and epoch_i < config.slow_start:
            train_iter = tqdm(shuffled_train_loader)
        else:
            train_iter = tqdm(train_loader)
        if epoch_i > 0:
            train_loader, test_loader = get_new_dataloader(real_vec=cl_real_vec, dtm_preds=cl_dtm_preds,
                                                           labels=cl_labels)
            train_iter = tqdm(train_loader)
        for batch_i, train_data in enumerate(train_iter):
            model.zero_grad()
            inputs = train_data["data"]
            labels = label_maker.get_labels(train_data["label"])
            cl_labels += labels
            outputs = model(inputs)
            cl_dtm_preds += label_maker.get_preds(inputs)
            cl_real_vec += inputs
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_iter.set_postfix({
                "train_loss": loss.item(),
                "best_presicion": best_precision,
                "best_recall": best_recall,
                "best_f1": best_f1
            })
        # evaluation
        precision, recall, f1 = evaluate(model, evaluate_loader, label_maker)
        if f1 > best_f1:
            torch.save(model.state_dict(), os.path.join(config.model_path, "model_F1_{}".format(round(f1, 5))))
        best_precision = max(precision, best_precision)
        best_recall = max(recall, best_recall)
        best_f1 = max(f1, best_f1)


def evaluate(model, evaluate_loader, label_maker):
    all_true_labels = []
    all_predicted_labels = []
    for test_data in evaluate_loader:
        model.zero_grad()
        inputs = test_data["data"]
        labels = labels = label_maker.get_labels(test_data["label"])
        outputs = model(inputs)
        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        predicted_labels = outputs.argmax(1).tolist()
        all_true_labels += list(labels)
        all_predicted_labels += predicted_labels
    # print(sum(all_true_labels))
    # print(sum(all_predicted_labels))
    precision = precision_score(all_true_labels, all_predicted_labels, average="macro")
    recall = recall_score(all_true_labels, all_predicted_labels, average="macro")
    f1 = f1_score(all_true_labels, all_predicted_labels, average="macro")
    return precision, recall, f1


def get_new_dataloader(cl, train_data, test_data, real_vec=None, dtm_preds=None, entropy_based=True, labels=None,
                       distance_based=True):
    train_orders = cl.get_next_epoch(train_data, real_vec, dtm_preds, entropy_based, labels, distance_based)
    test_orders = cl.get_next_epoch(test_data, real_vec, dtm_preds, entropy_based, labels, distance_based)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, orders=train_orders)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, drop_last=True, orders=test_orders)
    return train_loader, test_loader


if __name__ == '__main__':
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
    # train
    train_dataset = dataset_settings["swat"]["train_dataset"]
    test_dataset = dataset_settings["swat"]["test_dataset"]

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True)
    shuffled_train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, drop_last=True)
    # model = LSTMCNNModel(train_dataset.input_size, config)
    # model = CUSUMModel(train_dataset.input_size, config)
    # model = GANModel(train_dataset.input_size, config)
    model = ATTAINModel(config, train_dataset.input_size)
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    label_maker = LabelMaker()
    cl = CurriculumLearning(config)
    train(model, train_loader, test_loader, criterion, optimizer, cl, label_maker=label_maker,
          shuffled_train_loader=shuffled_train_loader)
