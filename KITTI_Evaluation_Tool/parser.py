#!/usr/bin/env python

import sys
import os
import numpy as np

# CLASSES = ['car', 'pedestrian', 'cyclist', 'van', 'truck', 'person_sitting', 'tram']
CLASSES = ['car', 'pedestrian', 'cyclist']

# PARAMS = ['detection', 'orientation', 'iour', 'mppe']
#PARAMS = ['detection', 'orientation']
PARAMS = ['detection']

DIFFICULTIES = ['easy', 'moderate', 'hard']

eval_type = ''

if len(sys.argv)<2:
    print('Usage: parser.py results_folder [evaluation_type]')

if len(sys.argv)==3:
    eval_type = sys.argv[2]

result_sha = sys.argv[1]
txt_dir = os.path.join('build','results', result_sha)

for class_name in CLASSES:
    for param in PARAMS:
        print("--{:s} {:s}--".format(class_name, param))
        if eval_type is '':
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '.txt')
        else:
            txt_name = os.path.join(txt_dir, 'stats_' + class_name + '_' + param + '_' + eval_type + '.txt')

        if not os.path.isfile(txt_name):
            print(txt_name, 'not found')
            continue

        cont = np.loadtxt(txt_name)
        # None if this class_name is detected.
        if cont.size == 0:
           continue

        averages = []
        for idx, difficulty in enumerate(DIFFICULTIES):
            sum = 0
            if (param is 'detection') or (param is 'orientation'):
                for i in range(1, 41):
                    sum += cont[idx][i]

                average = sum/40.0
            else:
                for i in range(0, cont.shape[1], 1):
                    sum += cont[idx][i]

                average = sum/cont.shape[1]
            averages.append(average)

        print("Easy\tMod.\tHard")
        print("{:.4f}\t{:.4f}\t{:.4f}".format(averages[0], averages[1], averages[2]))
        print("-----------------------\n")
        if eval_type is not '' and param=='detection':
            break # No orientation for 3D or bird eye

