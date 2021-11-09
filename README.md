# Digital Twin-based Anomaly Detection with Curriculum Learning in Cyber-physical Systems

# Abstract
Anomaly detection is critical to ensure the security of cyber-physical systems (CPS). However, due to theincreasing complexity and more sophisticated attacks, tackling the anomaly detection task in CPS is becomingmore and more challenging. In our previous work, we proposed a digital twin-based anomaly detection methodcalled ATTAIN, which takes advantage of both historical and real-time data of CPS. However, these samples canvary significantly in terms of difficulty, therefore, similar to human learning processes, deep learning models,i.e., ATTAIN, can benefit from an easy-to-difficult curriculum. To this end, we present a novel approach, calleddigitaL twin-based Anomaly deTecTion wIth Curriculum lEarning (LATTICE), by introducing curriculumlearning to optimize the learning paradigm of ATTAIN. LATTICE uses a difficulty scorer to assign scores foreach sample, which is then fed into a training scheduler. The training scheduler samples batches of trainingdata based on these scores to perform learning from easy to difficult data. To evaluate our approach, we usedfive publicly available datasets collected from five CPS testbeds. We compare LATTICE with ATTAIN and twoother state-of-the-art anomaly detectors. Evaluation results show that LATTICE improves the performance of the baselines by 0.906%-2.367% in terms of F1 score.
# Overview

# Installation

# Dataset
