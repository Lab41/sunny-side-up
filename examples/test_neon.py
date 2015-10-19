#!/usr/bin/env python
"""
Demo character-level CNN on Neon
"""

import sys
import os
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_path))

import neon

import src.datasets.amazon_reviews as amazon
from src.datasets.batch_data import batch_data as batcher

def get_data():
    a = batcher(amazon.load_data("/root/data/amazon/test_amazon.json.gz"))
    return a


def main():
    a = get_data()
    #print a.next()

if __name__=="__main__":
    main()


