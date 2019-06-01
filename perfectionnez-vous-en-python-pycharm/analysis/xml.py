#! usr/bin/env python3
#coding: utf-8

import os

def launch_analysis(data_file):
    directory = os.path.dirname(os.path.dirname(__file__))
    path_to_file = os.path.join(directory,"data",data_file)

    with open(path_to_file) as f:
        preview = f.readline()

    print(f'Yeah we have loaded the file starting with {preview}')

if __name__ == "__main__":
    launch_analysis("SyceronBrut.xml")