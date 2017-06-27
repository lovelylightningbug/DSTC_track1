#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 14:28:46 2017

@author: suman
"""


import numpy as np
import random
import json
import unicodedata

from sklearn.feature_extraction.text import TfidfVectorizer


def do_parse_cmdline():

    from optparse import OptionParser
    parser = OptionParser()

    parser.add_option("--input-task-file", dest="inputtaskfile",
                      default="dialog-task1API-kb1_atmosphere-distr0.5-trn10000.json",
                      help="filename of the task", metavar="FILE")

    parser.add_option("--output_result-file", dest="outputresultfile",
                      default="output-result-tfidf.json",
                      help="output file results", metavar="FILE")

    (options, args) = parser.parse_args()

    return options.inputtaskfile, options.outputresultfile


if __name__ == '__main__':

    # Parsing command line
    inputtaskfile, outputresultfile = do_parse_cmdline()

    fd = open(inputtaskfile, 'rb')
    json_data = json.load(fd)
    fd.close()
    
    output_truth_file='output-truth.json'
    
    lst_truth = []
    for idx_story, story in enumerate(json_data):
        dict_answer_current = {}
        dict_answer_current['dialog_id'] = story['dialog_id']
        lst_candidate_id   = []
        a=story['answer']['candidate_id']
        lst_candidate_id.append({'candidate_id':a,'rank':1})
        
        dict_answer_current['lst_candidate_id'] = lst_candidate_id
        lst_truth.append(dict_answer_current)                       

    fd_out = open(output_truth_file, 'wb')
    json.dump(lst_truth, fd_out)
    fd_out.close()

    