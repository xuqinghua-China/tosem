
from settings import Config
import pandas as pd
import picklec
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import is_actuator_name, remove_outlier, count_frequency, hamming_distance


class StateVocabulary:
    """maintaining all the states"""

    def __init__(self):
        self.idx2state = []
        self.state2idx = {}
        self.freq = {}
        self.prob = {}
        self.total = 0

    def add(self, state):
        self.idx2state.append(state)
        self.state2idx[state] = len(self.state2idx)
        self.freq[state] = 0
        self.prob[state] = 0
        self.total += 1

    def get_id(self, state):
        return self.state2idx[state]

    def exists(self, state):
        return state in self.state2idx

    def get_id_or_add(self, state):
        if not self.exists(state):
            self.add(state)
        return self.get_id(state)

    def update_prob(self, state):
        self.freq[state] += 1
        self.total += 1
        self.prob[state] = self.freq[state] / self.total

    def get_prob(self, state):
        return self.prob[state]


class PairedStateVocabulary:
    """maintaining paired state vocabulary"""

    def __init__(self):
        self.idx2pairedstate = []
        self.pairedstate2idx = {}
        self.freq = {}
        self.prob = {}
        self.total = 0

    def get_key(self, s1, s2):
        return "_".join([s1, s2])

    def add(self, s1, s2):
        key = self.get_key(s1, s2)
        self.idx2pairedstate.append(key)
        self.pairedstate2idx[key] = len(self.pairedstate2idx)
        self.freq[key] = 1
        self.total += 1
        self.prob[key] = self.freq[key] / self.total

    def get_id(self, s1, s2):
        key = self.get_key(s1, s2)
        return self.pairedstate2idx[key]

    def exists(self, s1, s2):
        key = self.get_key(s1, s2)
        return key in self.pairedstate2idx

    def update_or_add(self, s1, s2):
        if not self.exists(s1, s2):
            self.add(s1, s2)
        else:
            key = self.get_key(s1, s2)
            self.freq[key] += 1
            self.total += 1
            self.prob[key] = self.freq[key] / self.total

    def get_prob(self, s1, s2):
        key = self.get_key(s1, s2)
        return self.prob[key]

    def max_prob(self, src):
        max_prob = -1
        max_tgt = None
        for pair, prob in self.prob.items():
            if pair.startswith(src + "_") and prob > max_prob:
                max_prob = prob
                max_tgt = tgt
        return max_prob, max_tgt


class TimedTransitionVocabulary:
    """maintaining all transition probability"""

    def __init__(self, timing, data):
        self.timing = timing
        self.state_vocab = StateVocabulary()
        self.probability_table = {t: PairedStateVocabulary for t in timing}
        self.data = data

    def update_probability(self, transition_t, tgt):
        tgt_idx = self.state_vocab.get_id_or_add(tgt)
        self.state_vocab.update_prob(tgt)
        srcs = {t: self.data[transition_t - t] for t in self.timing}
        for t, src_t in srcs.items():
            src_state = self.data[src_t]
            src_idx = self.state_vocab.get_id_or_add(src_state)
            self.probability_table[t].update_or_add(src_idx, tgt_idx)

    def max_prob(self, src):
        max_prob = -1
        max_tgt = None
        for t, pairedStateVocabulary in self.probability_table.items():
            vocab_max_prob, vocab_max_tgt = pairedStateVocabulary.max_prob(src)
            if vocab_max_prob > max_prob:
                max_prob = vocab_max_prob
                max_tgt = vocab_max_tgt

        return max_prob, max_tgt


class TimedAutomata:
    """Timed automaton machine, updating states, transitions, time constraints with OTALA"""

    def __init__(self, data):
        self.S = StateVocabulary()
        self.T = set()
        self.timing = list(range(0, 100, 10))
        self.timed_transition_vocab = TimedTransitionVocabulary(self.timing, data)

    def observe(self, u):
        transition_t, tgt = u
        self.timed_transition_vocab.update_probability(transition_t, tgt)

    def predict(self, src):
        return self.timed_transition_vocab.max_prob(src)

    def get_ground_truth_label(self, u, real, threshhold):
        predicted_prob, predicted_state = self.timed_transition_vocab.max_prob(u)
        return hamming_distance(real, predicted_state) < threshhold


if __name__ == '__main__':
    # calculate time interval changing
    config = Config()
    swat_train = pickle.load(open(os.path.join(config.pickle_path, "swat_train.pkl"), "rb"))
    swat_test = pickle.load(open(os.path.join(config.pickle_path, "swat_test.pkl"), "rb"))
    if os.path.exists(os.path.join(config.pickle_path, "swat_idx.pkl")):
        states_list = pickle.load(open(os.path.join(config.pickle_path, "swat_idx.pkl"), "rb"))
    else:

        swat = swat_train.append(swat_test)
        actuator_names = [col for col in swat.columns if is_actuator_name(col)]
        actuator_df = swat[actuator_names]
        actuator_df = actuator_df.astype(str)
        actuator_df["state"] = ["" for _ in range(actuator_df.shape[0])]
        for col in actuator_names:
            actuator_df["state"] += actuator_df[col]
        states_list = actuator_df["state"].to_list()
        pickle.dump(states_list, open(os.path.join(config.pickle_path, "swat_idx.pkl"), "wb"))
    tm = TimedAutomata(states_list)
    unique_states = set(states_list)
    print(len(unique_states))
    print(len(states_list))

    # pretrain automata
    for state in states_list:
        tm.observe(state)

    # save
    pickle.dump(tm, open(os.path.join(config.pickle_path, "tm.pkl"), "wb"))
    # calculate time intervals
    analysed = True
    if not analysed:
        time_intervals = []
        timer = 0
        for i in range(len(states_list) - 1):
            src = states_list[i]
            tgt = states_list[i + 1]
            if src != tgt:
                timer += 1
                time_intervals.append((src, tgt, timer))
                timer = 0
            else:
                timer += 1
        time_interval_values = [time_interval[2] for time_interval in time_intervals]
        # get min/max/frequency
        print("min:", min(time_interval_values))
        print("max:", max(time_interval_values))
        print("mean", sum(time_interval_values) / len(time_interval_values))
        print("median", np.quantile(time_interval_values, 0.5))
        time_interval_values = remove_outlier(time_interval_values)
        freqs = count_frequency(time_interval_values)
        print(freqs)
        plt.bar(freqs.keys(), freqs.values())
        plt.show()

    # pretrain
