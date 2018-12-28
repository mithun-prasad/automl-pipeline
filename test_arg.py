import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-ws.name','--filename1', default='example.csv')
args = vars(parser.parse_args())
print(args['filename1'])

