"""
Author: xuqh
Created on 2021/5/11
"""
import pickle
import numpy as np
from enums import DifficultyScorerStrategy
from random import random

class CurriculumLearning:
    def __init__(self, config):
        self.ds = DifficultyScorer(config)
        self.ts = TrainingScheduler(config)

    def get_next_epoch(self, data, real_vec=None, dtm_preds=None, entropy_based=True, labels=None, distance_based=True):
        scores = self.ds.get_score(data)
        epoch_ids = self.ts.baby_step(scores)
        return epoch_ids


class DifficultyScorer:
    def __init__(self, config, attack_log_path=None, strategy=DifficultyScorerStrategy.HYBRID):
        self.attack_log = pickle.load(attack_log_path, "rb")  # start index & end index
        self.strategy = strategy
        self.config = config

    def get_score(self, data, real_vec=None, dtm_preds=None, entropy_based=True, labels=None, distance_based=True):
        if self.strategy == DifficultyScorerStrategy.STATIC:
            return self.score_static(data)
        elif self.strategy == DifficultyScorerStrategy.DYNAMIC:
            return self.score_dynamic(data)
        elif self.strategy == DifficultyScorerStrategy.HYBRID:
            return self.config.hybrid_difficulty_ratio * self.score_static(data) + (
                    1 - self.config.hybrid_difficulty_ratio) * self.score_dynamic(data)

    def score_static(self, ids):
        distances = [self.calculate_distance(id) for id in ids]
        return self.rescale(distances)

    def score_dynamic(self, real_vec=None, dtm_preds=None, entropy_based=True, labels=None, distance_based=True):
        if entropy_based is True and distance_based is False:
            entropy_score = [self.calculate_entropy(yHat, y) for yHat, y in enumerate(labels)]
            return self.rescale(entropy_score)
        if entropy_based is False and distance_based is True:
            distance_score = [self.claculate_distance(x, y) for x, y in enumerate(real_vec, dtm_preds)]
            return self.rescale(distance_score)
        if entropy_based is True and distance_based is True:
            entropy_score = [self.calculate_entropy(yHat, y) for yHat, y in enumerate(labels)]
            distance_score = [self.claculate_distance(x, y) for x, y in enumerate(real_vec, dtm_preds)]
            combined = entropy_based + distance_based
            return self.rescale(combined)

    def calculate_distance(self, id):
        attack_log = self.attack_log
        id = self.find_closest_id(id)
        closest_id = self.find_closest_id(id)
        closest_start, closest_end = attack_log[closest_id]
        distance = (id - closest_start) / (closest_end - closest_start)
        return distance

    def calculate_entropy(self, yHat, y):
        if y == 1:
            return -np.log(yHat)
        else:
            return -np.log(1 - yHat)

    def claculate_distance(self, x, y):
        ans = 0
        for i in range(31, -1, -1):
            b1 = x >> i & 1
            b2 = y >> i & 1
            ans += not (b1 == b2)
        return ans

    def find_closest_id(self, idx):
        attack_log = self.attack_log
        start_time = [start_time for start_time, end_time in attack_log]
        end_time = [end_time for start_time, end_time in attack_log]
        start_difference = [abs(idx - start_time) for i in start_time]
        end_difference = [abs(idx - end_time) for i in start_time]
        closest_start = start_difference.index(min(start_difference))
        closest_end = end_difference.index(min(end_difference))
        if min(start_difference) > min(end_difference):
            return closest_end
        else:
            return closest_start

    def rescale(self, data):
        maximum, minimum = max(data), min(data)
        data = np.array(data)
        return (data - minimum) / (maximum) / minimum


class TrainingScheduler:
    def __init__(self, config, scores, single_sample_size=4):
        self.batch_size = config.batch_size
        self.groups = [i % single_sample_size for i in scores]
        self.single_sample_size = single_sample_size
        self.n_epochs = config.n_epochs
        self.current_epoch = 1

    def baby_step(self, scores):
        ids = np.argsort(scores)
        group_scores = [sum(scores[i:i + self.single_sample_size]) / self.single_sample_size for i in
                        range(0, scores, self.single_sample_size) / self.single_sample_size]
        n_batches = len(scores) / self.batch_size
        n_easy = n_difficulty = n_batches / 2  # to explore more in the future
        easy_ratio = 1.0 / self.current_epoch * self.config.baby_step_ratio
        difficulty_ratio = 1 - easy_ratio
        easy_ids = random.sample(ids[:n_easy], easy_ratio * len(scores))
        difficult_ids = random.sample(ids[n_easy:], difficulty_ratio * len(scores))
        epoch_ids = easy_ids + difficulty_ids
        self.current_epoch += 1
        return epoch_ids
