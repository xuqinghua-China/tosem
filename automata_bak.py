"""
Author: xuqh
"""

import pickle
import numpy as np

from utils import hm

MAX_N_STATE = 10000
MAX_TIME_CONSTRAINTS = 10


class StateVocabulary:
    """maintaining all the states"""

    def __init__(self):
        self.idx2state = []
        self.state2idx = {}

    def observation2state(self, observation):
        return observation

    def add(self, state):
        self.idx2state.append(state)
        self.state2idx[state] = len(self.state2idx)

    def get_id(self, state):
        return self.state2idx[state]

    def exists(self, state):
        return state in self.state2idx

    def get_id_or_add(self, state):
        if not self.exists(state):
            self.add(state)
        return self.get_id(state)

    def get_state(self, id):
        return self.idx2state[id]


class HistoryState:
    """preserve history state within a window"""

    def __init__(self):
        self.history_state = []

    def add_state(self, state):
        if len(self.history_state) >= MAX_TIME_CONSTRAINTS:
            self.history_state = self.history_state[1:] + [state]
        else:
            self.history_state.append(state)


class TransitionVocabulary:
    """maintaining all transitions"""

    def __init__(self):
        self.transition = [[np.zeros(MAX_TIME_CONSTRAINTS) for j in range(MAX_N_STATE)] for i in range(MAX_N_STATE)]
        self.transition_pdf = [[np.zeros(MAX_TIME_CONSTRAINTS) for j in range(MAX_N_STATE)] for i in range(MAX_N_STATE)]

    def update_transition(self, src_id, tgt_id, n_time_interval):
        self.transition[src_id][tgt_id][n_time_interval - 1] += 1
        self.transition_pdf[src_id][tgt_id] = self.transition[src_id][tgt_id] / np.sum(self.transition[src_id][tgt_id])


class TimedAutomata:
    def __init__(self, config=None):
        self.stateVocab = StateVocabulary()
        self.transitionVocab = TransitionVocabulary()
        self.historyState = HistoryState()
        self.config = config
        init_state = "INIT"
        self.stateVocab.add(init_state)
        self.historyState.add_state(self.stateVocab.get_id(init_state))

    def train(self, observation):
        state = self.stateVocab.observation2state(observation)
        self.stateVocab.add(state)
        state = self.stateVocab.get_id(state)
        self.historyState.add_state(state)
        for i, history_state in enumerate(self.historyState.history_state):
            self.transitionVocab.update_transition(history_state, state, MAX_TIME_CONSTRAINTS - i)

    def predict(self):
        predicted_tgt = None
        predicted_prob = -1
        for i, history_state in enumerate(self.historyState.history_state):

            probs = np.array([
                tgts[MAX_TIME_CONSTRAINTS - i - 1]
                for tgts in self.transitionVocab.transition_pdf[history_state]

            ])
            max_prob_tgt = np.argmax(probs)
            max_prob = probs[max_prob_tgt]
            if max_prob > predicted_prob:
                predicted_prob = max_prob
                predicted_tgt = max_prob_tgt
        predicted_tgt = self.stateVocab.get_state(predicted_tgt)
        return predicted_tgt, predicted_prob

    def batch_predict(self, batched_data):
        predicted = []
        for data in batched_data:
            predicted_tgt, predicted_prob = self.predict()
            predicted.append((predicted_tgt, predicted_prob))
            self.train(data)
        return predicted

    def batch_predict_stateless(self, batched_data):
        old_historyState = self.historyState
        predicted = self.batch_predict(batched_data)
        self.historyState = old_historyState
        return predicted

    def gtl(self, batached_data, real_data):
        predicted = self.batch_predict_stateless(batached_data)
        s = 0
        for d, r_d in zip(predicted, real_data):
            dist = hm(d, r_d)
            s += dist
        return self.is_at(s / len(batached_data))

    def is_at(self, d):
        th = self.config.hmt
        return d > th


if __name__ == '__main__':
    fpath = "dataset/swat_idx.pkl"
    data = pickle.load(open(fpath, "rb"))
    ta = TimedAutomata()

    for observation in data:
        ta.train(observation)
        predicted_tgt, _ = ta.predict()
        print(predicted_tgt)
