import numpy as np
import torch
from torch.nn import CrossEntropyLoss
import torch


def is_actuator_name(name):
    name = name.upper()
    if "IT" in name:
        return False
    if name.startswith("MV"):
        return True
    if name.startswith("DPIT"):
        return True
    if name.startswith("P"):
        return True
    return False


def remove_outlier(data):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    lower_limit = q1 - 1.5 * iqr
    data = [elem for elem in data if lower_limit <= elem <= upper_limit]
    return data


def count_frequency(data):
    freqs = {}
    for d in data:
        if d not in freqs:
            freqs[d] = 0
        else:
            freqs[d] += 1
    return freqs


def hamming_distance(u1, u2):
    return sum(c1 != c2 for c1, c2 in zip(u1, u2))


def gan_loss(real_outputs, fake_outputs):
    criterion = CrossEntropyLoss()
    real_loss = torch.mean(real_outputs, torch.ones_like(real_outputs))
    fake_loss = torch.mean(fake_outputs, torch.zeros_like(fake_outputs))
    d_loss = real_loss + fake_loss
    g_loss = torch.mean(fake_outputs, torch.ones_like(fake_outputs))
    return g_loss, d_loss


def rearrange_data():
    """rearrange data order to increase loss at beginning"""


class EdgeMatrix:
    def __init__(self, n_indices):
        self.n_indices = n_indices

    def get_all_connected_indices(self):
        n_indices = self.n_indices
        edge_index_list = [
            [],
            []
        ]
        for src in range(n_indices):
            for tgt in range(n_indices):
                if src == tgt:
                    continue
                edge_index_list[0].append(src)
                edge_index_list[1].append(tgt)
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
        return edge_index_tensor


class LabelMaker:
    def __init__(self, n_classes=3, autamata=None):
        self.n_classes = n_classes
        self.autamata = autamata

    def get_labels(self, labels, fake_data=None, real_data=None):
        if self.n_classes == 2:
            return self.get_2class_labels(labels)
        if self.n_classes == 3:
            return self.get_3class_labels(labels)
        if self.n_classes == 4:
            return self.get_4class_labels(labels, fake_data)

    def get_2class_labels(self, real_labels):
        fake_labels = torch.ones_like(real_labels)
        real_labels = torch.zeros_like(real_labels)
        return torch.cat([real_labels, fake_labels])

    def get_3class_labels(self, real_labels):
        fake_labels = torch.ones_like(real_labels)
        # fake_labels += fake_labels
        fake_labels = torch.ones_like(real_labels)
        labels = torch.cat([real_labels, fake_labels])
        return labels

    def get_4class_labels(self, real_labels, fake_data, real_data):
        automata = self.autamata
        predicted = automata.batch_predict_stateless(fake_data)
        # todo data transformation
        predicted_state = [state for state, _ in predicted]
        fake_labels = [compare_hamming_distance(state) for state in predicted_state]
        fake_labels = np.array(fake_labels)
        labels = torch.cat([real_labels, fake_labels])
        automata.batch_predict(real_data)
        return labels
    def get_preds(self,data):
        return self.automata.batch_predict_stateless(data)

def compare_hamming_distance(state1, state2):
    hd = hm(state1, state2)
    threshhold = 2
    label = 1 if hd > threshhold else 0
    return label


def hm(x, y):
    """
    :type x: int
    :type y: int
    :rtype: int
    """
    ans = 0
    for i in range(31, -1, -1):
        b1 = x >> i & 1
        b2 = y >> i & 1
        ans += not (b1 == b2)
        # if not(b1==b2):
        # print(b1,b2,i)
    return ans


def vec2state(vec):
    return "".join(vec)


def find_diff(s1, s2):
    s1 = set(s1)
    s2 = set(s2)
    diff = None
    if len(s1) > len(s2):
        diff = s1 - s2
    else:
        diff = s2 - s1
    return "Missing: {}".format(",".join(diff))


if __name__ == '__main__':
    u1 = "abcde"
    u2 = "ddcde"
    print(hamming_distance(u1, u2))

    # test EdgeMatrix
    edge_matrix = EdgeMatrix(10)
    print(edge_matrix.get_all_connected_indices())
