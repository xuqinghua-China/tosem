
# Digital Twin-based Anomaly Detection with Curriculum Learning in Cyber-physical Systems

# Abstract
Anomaly detection is critical to ensure the security of cyber-physical systems (CPS). However, due to theincreasing complexity and more sophisticated attacks, tackling the anomaly detection task in CPS is becomingmore and more challenging. In our previous work, we proposed a digital twin-based anomaly detection methodcalled ATTAIN, which takes advantage of both historical and real-time data of CPS. However, these samples canvary significantly in terms of difficulty, therefore, similar to human learning processes, deep learning models,i.e., ATTAIN, can benefit from an easy-to-difficult curriculum. To this end, we present a novel approach, calleddigitaL twin-based Anomaly deTecTion wIth Curriculum lEarning (LATTICE), by introducing curriculumlearning to optimize the learning paradigm of ATTAIN. LATTICE uses a difficulty scorer to assign scores foreach sample, which is then fed into a training scheduler. The training scheduler samples batches of trainingdata based on these scores to perform learning from easy to difficult data. To evaluate our approach, we usedfive publicly available datasets collected from five CPS testbeds. We compare LATTICE with ATTAIN and twoother state-of-the-art anomaly detectors. Evaluation results show that LATTICE improves the performance of the baselines by 0.906%-2.367% in terms of F1 score.
# Overview
![overview](https://user-images.githubusercontent.com/62027704/141008868-0220f42b-1dcb-4791-9f0a-57fa7b641118.png)
LATTICE follows the general CL framework proposed in [28], as we discussed in Section 2, withthe main idea of training models from easier data to harder data. Therefore, a general CL designconsists ofDifficulty MeasurerandTraining Scheduler, which, respectively, decide the relative"easiness" of each sample and the sequence of data subsets throughout the training process basedon the judgment ofDifficulty Measurer. As shown in Figure 3, all the training examples are sortedby theDifficulty Measurerfrom the easiest to the hardest and passed to theTraining Scheduler. Then, at each training epoch𝑡, theTraining Schedulersamples a batch of training data from therelatively easier examples, and sends it to theATTAINfor training. With progressing trainingepochs, theTraining Schedulerdecides when to sample from harder data. As shown in the runningexample (Table 1), the Difficulty column presents the difficulty scores given by theDifficultyMeasurer, indicating the relative "easiness" of each sample. For instance, the sample data at 10:00:00is assigned a difficulty score of 0.9, i.e,𝑠𝑐𝑜𝑟𝑒(𝑢10:00:00)=0.9, while the difficulty score of the sampleat 10:29:12 is 0.5, i.e,𝑠𝑐𝑜𝑟𝑒(𝑢10:29:12)=0.5. This tells that sample𝑢10:00:00is relatively harder for themodel to learn. With these difficulty scores, the training scheduler decides which samples shouldbe included in each batch. The general principle is that easy samples should be included first. Aftercalculation, the training scheduler assigns new batch numbers for𝑢10:00:00(batch number=23) and𝑢10:19:12(batch number=2). In the following section, we will present more details about theDifficultyMeasurer,Training Schedulerand the extension to ATTAIN in Section 5.1, Section 5.2, and Section5.3, respectively
# Dataset
![dataset](https://user-images.githubusercontent.com/62027704/141009105-e7cbee65-c6f6-48e6-9ea6-6ff6c2596132.png)

# Train
