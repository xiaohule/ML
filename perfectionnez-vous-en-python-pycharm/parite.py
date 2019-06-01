#! /usr/bin/env python3
#coding: utf-8

import argparse
import logging as lg

import analysis.csv as c_an
import analysis.xml as x_an

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--extension", help="""Type of file to analyse. Is it CSV or XML?""")
    parser.add_argument("-f", "--file", help="""Path to the file to analyse.""")
    return parser.parse_args()

def main():
    args = parse_argument()
    file = args.file
    # import pdb; pdb.set_trace()
    try:
        if args.extension == 'csv':
            c_an.launch_analysis(file)
        elif args.extension == 'xml':
            x_an.launch_analysis(file)
    except TypeError as e:
        lg.warning(f"Missing file, please specify data file with --file option \n {e}")

if __name__ == "__main__":
    main()