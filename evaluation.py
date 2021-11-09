import os
import pickle
import pandas as pd

from automata import TimedAutomata
from settings import Config
import matplotlib.pyplot as plt

from utils import hamming_distance


def outlier_detection(model, dataset, threshhold):
    count = 0
    for real, label in dataset.items():
        predicted = model.predict()
        distance = hamming_distance(real, predicted)
        predicted_label = 0
        if distance > threshhold:
            predicted_label = 1
        if predicted_label == label:
            count += 1
    accuracy = count / len(dataset)
    return accuracy


if __name__ == '__main__':
    # draw hamming distance
    config = Config()
    data = pickle.load(open(os.path.join(config.model_path, "hamming.pkl")))
    x1, y1 = data["x1"], data["y1"]
    x2, y2 = data["x2"], data["y2"]
    x3, y3 = data["x3"], data["y3"]
    l1 = plt.plot(x1, y1, marker="o", label='SWAT')
    l2 = plt.plot(x1, y2, marker="x", label='WADI')
    l3 = plt.plot(x1, y3, marker="s", label='BATADAL')
    plt.title('Average Hamming distance of Digital Twin Model Predictions')
    plt.xlabel('# of samples(Thousand)')
    plt.ylabel('Hamming distance')
    plt.legend()
    plt.show()
    # calculate outlier detection rate
    model = pickle.load(open(os.path.join(config.model_path, "automata.pkl"), "rb"))
    swat_data = pickle.load(open(os.path.join(config.model_path, "automata.pkl"), "rb"))
    wadi_data = pickle.load(open(os.path.join(config.model_path, "automata.pkl"), "rb"))
    batadal_data = pickle.load(open(os.path.join(config.model_path, "automata.pkl"), "rb"))
    swat_accuracy = outlier_detection(model, swat_data, config.threshhold)
    wadi_accuracy = outlier_detection(model, wadi_data, config.threshhold)
    batadal_accuracy = outlier_detection(model, batadal_data, config.threshhold)
    avg_accuracy = (swat_accuracy + wadi_accuracy + batadal_accuracy) / 3
    print("Average outlier detection accuracy is:", avg_accuracy)

    # calculate anomaly detection rate in  train.py
