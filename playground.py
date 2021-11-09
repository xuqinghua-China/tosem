import numpy as np

complexity = np.array([0.436, 0.771, 0.472, 0.180, 0.479])
lstm_cusum = np.array([1.330, 3.779, 2.145, 0.890, 2.530])
mad_gan = np.array([3.762, 11.027, 3.908, 0.996, 4.115])
attain = np.array([3.844, 12.013, 3.880, 1.101, 4.230])
lattice = np.array([3.712, 9.355, 3.771, 1.537, 4.024])
unit_lstm_cusum = lstm_cusum / complexity
unit_mad_gan = mad_gan / complexity
unit_attain = attain / complexity
unit_lattice = lattice / complexity
for m1, m2, m3, m4 in zip(unit_lstm_cusum, unit_mad_gan, unit_attain, unit_lattice):
    print("{} & {} & {} & {} \\\\ \\hline".format(m1.round(3), m2.round(3), m3.round(3), m4.round(3)))

table = """
    & Precision &0.90701 & 0.96136 & 0.95994 & 0.96112 & 0.97096
    & Recall & 0.67721& 0.94228  &0.99231 & 0.98142 & 0.98992
    & F1 &0.77544 & 0.95172 & 0.97585 & 0.97116 & 0.98491
    & Precision &0.61302 & 0.43212 & 0.66586 & 0.64992 & 0.69787 
    & Recall & 0.69763 & 0.95274 & 0.84411 & 0.85011 & 0.85402
    & F1 &0.65959&0.59456 & 0.74446 & 0.73666 & 0.76809
    & Precision &0.65773 &0.52991 &0.72238 & 0.72847 & 0.76357
    & Recall &0.72105 &0.96213 &0.76341 & 0.76021 & 0.77551 
    & F1 &0.68794 & 0.68342 &0.74232 & 0.74400 & 0.76949
    & Precision &0.85421 &0.81722 &0.88021 & 0.86903 & 0.89945
    & Recall &0.80877 &0.89306 &0.90010 & 0.89114 & 0.92106
    & F1 &0.83087 &0.85373 &0.89004 & 0.87995 & 0.91012
    & Precision &0.71532 &0.79207 &0.81962 & 0.83044 & 0.84141
    & Recall &0.72149 &0.86690 &0.85413 & 0.86237 & 0.88405
    & F1 &0.71834 & 0.82780&0.83652 & 0.84610 & 0.86220
"""
lines = table.split("\n")
results=[]
for line in lines:
    if line == '':
        continue
    line = line.strip()
    nums=line.split("&")
    real_nums=[]
    for num in nums:
        num=num.strip()
        if num.strip().replace('.','',1).isdigit():
            real_nums.append(float(num))
    results.append(real_nums)



lc_precision=[results[0][0],results[3][0],results[6][0],results[9][0],results[12][0]]
attain_precision=[results[0][2],results[3][2],results[6][2],results[9][2],results[12][2]]
lattice_s1_precision=[results[0][3],results[3][3],results[6][3],results[9][3],results[12][3]]
lattice_s2_precision=[results[0][4],results[3][4],results[6][4],results[9][4],results[12][4]]
avg_lc_precision=sum(lc_precision)/len((lc_precision))
avg_attain_precision=sum(attain_precision)/len(attain_precision)
avg_lattice_s1_precision=sum(lattice_s1_precision)/len(lattice_s1_precision)
avg_lattice_s2_precision=sum(lattice_s2_precision)/len(lattice_s2_precision)
print(avg_attain_precision,avg_lattice_s1_precision,avg_lattice_s2_precision)

lc_recall=[results[1][0],results[4][0],results[7][0],results[10][0],results[13][0]]
attain_recall=[results[1][2],results[4][2],results[7][2],results[10][2],results[13][2]]
lattice_s1_recall=[results[1][3],results[4][3],results[7][3],results[10][3],results[13][3]]
lattice_s2_recall=[results[1][4],results[4][4],results[7][4],results[10][4],results[13][4]]
avg_lc_recall=sum(lc_recall)/len((lc_recall))
avg_attain_recall=sum(attain_recall)/len(attain_recall)
avg_lattice_s1_recall=sum(lattice_s1_recall)/len(lattice_s1_recall)
avg_lattice_s2_recall=sum(lattice_s2_recall)/len(lattice_s2_recall)
print(avg_attain_recall,avg_lattice_s1_recall,avg_lattice_s2_recall)

lc_f1=[results[2][0],results[5][0],results[8][0],results[11][0],results[14][0]]
attain_f1=[results[2][2],results[5][2],results[8][2],results[11][2],results[14][2]]
lattice_s1_f1=[results[2][3],results[5][3],results[8][3],results[11][3],results[14][3]]
lattice_s2_f1=[results[2][4],results[5][4],results[8][4],results[11][4],results[14][4]]
avg_lc_f1=sum(lc_f1)/len((lc_f1))
avg_attain_f1=sum(attain_f1)/len(attain_f1)
avg_lattice_s1_f1=sum(lattice_s1_f1)/len(lattice_s1_f1)
avg_lattice_s2_f1=sum(lattice_s2_f1)/len(lattice_s2_f1)
print(avg_attain_f1,avg_lattice_s1_f1,avg_lattice_s2_f1)
