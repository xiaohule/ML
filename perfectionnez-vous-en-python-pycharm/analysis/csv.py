#! /usr/bin/env python3
#coding: utf-8

import os

def launch_analysis(data_file):

    directory = os.path.dirname(os.path.dirname(__file__))
    path_to_file = os.path.join(directory,"data",data_file)

    with open(path_to_file,'r') as f:
        preview = f.readline()

    print(f'Yeah! We managed to read the file. Here is a preview: {preview}')

if __name__ == "__main__":
    launch_analysis('nosdeputes.fr_deputes_en_mandat_2019-06-01.csv')