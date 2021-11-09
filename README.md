
# Digital Twin-based Anomaly Detection with Curriculum Learning in Cyber-physical Systems

# Abstract
Anomaly detection is critical to ensure the security of cyber-physical systems (CPS). However, due to the increasing complexity and more sophisticated attacks, tackling the anomaly detection task in CPS is becoming more and more challenging. In our previous work, we proposed a digital twin-based anomaly detection method called ATTAIN, which takes advantage of both historical and real-time data of CPS. However, these samples canvary significantly in terms of difficulty, therefore, similar to human learning processes, deep learning models,i.e., ATTAIN, can benefit from an easy-to-difficult curriculum. To this end, we present a novel approach, called digitaL twin-based Anomaly deTecTion wIth Curriculum lEarning (LATTICE), by introducing curriculum learning to optimize the learning paradigm of ATTAIN. LATTICE uses a difficulty scorer to assign scores for each sample, which is then fed into a training scheduler. The training scheduler samples batches of training data based on these scores to perform learning from easy to difficult data. To evaluate our approach, we used five publicly available datasets collected from five CPS testbeds. We compare LATTICE with ATTAIN and two other state-of-the-art anomaly detectors. Evaluation results show that LATTICE improves the performance of the baselines by 0.906%-2.367% in terms of F1 score.
# Overview
![overview](https://user-images.githubusercontent.com/62027704/141008868-0220f42b-1dcb-4791-9f0a-57fa7b641118.png)

LATTICE follows the general CL framework , with the main idea of training models from easier data to harder data. Therefore, a general CL design consists of Difficulty Measurerand Training Scheduler, which, respectively, decide the relative"easiness" of each sample and the sequence of data subsets throughout the training process based on the judgment of Difficulty Measurer. As shown in the figure above, all the training examples are sorted by the Difficulty Measurer from the easiest to the hardest and passed to the Training Scheduler. Then, at each training epoch, the Training Scheduler samples a batch of training data from the relatively easier examples, and sends it to the ATTAIN for training. With progressing training epochs, the Training Scheduler decides when to sample from harder data. 
# Dataset
![dataset](https://user-images.githubusercontent.com/62027704/141009105-e7cbee65-c6f6-48e6-9ea6-6ff6c2596132.png)

Due to copyright issues, we can not include public dataset in this repository. The references of the datasets are provided as follows.
1. **SWaT.** Mathur AP, Tippenhauer NO. SWaT: a water treatment testbed for research and training on ICS security. In: 2016 International Workshop on Cyber-physical Systems for Smart Water Networks (CySWater). IEEE; 2016. p. 31–6. 
2. **WADI.** Ahmed CM, Palleti VR, Mathur AP. WADI: a water distribution testbed for research in the design of secure cyber physical systems. In: Proceedings of the 3rd International Workshop on Cyber-Physical Systems for Smart Water Networks. 2017. p. 25–8. 
3. **BATADAL.** Taormina R, Galelli S, Tippenhauer NO, Salomons E, Ostfeld A, Eliades DG, et al. The Battle Of The Attack Detection Algorithms: Disclosing Cyber Attacks On Water Distribution Networks. J Water Resour Plan Manag. 2018 Aug;144(8):4018048. 
4. **PHM2015 Challenge.** Xiao W. A probabilistic machine learning approach to detect industrial plant faults: PHM15 data challenge. Proc Annu Conf Progn Heal Manag Soc PHM. 2015;(c):718–26. 
5. **Gas Pipeline Dataset.** Morris TH, Thornton Z, Turnipseed I. Industrial Control System Simulation and Data Logging for Intrusion Detection System Research. Seventh Annu Southeast Cyber Secur Summit [Internet]. 2015;6. 
# Train
### Environment Installation
Please install required python packages before you run any files. The command for installation is as follows.
```bash
pip install -r requirements.txt
```
