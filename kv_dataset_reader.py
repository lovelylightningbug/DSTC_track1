#!/usr/bin/python

import argparse
import csv
import sys
import random

import numpy as np

#from data_utils import *

from collections import defaultdict
from tqdm import tqdm
import json


SPACE=" "

def get_maxlen(*paths):
  maxlen = defaultdict(int)
  for path in paths:
    with open(path, 'r') as examples_file:
      reader = json.load(examples_file)
      for row in reader:
        example = {}
        example['ans'] = row['ans']
        example['ans_candidates'] = row['ans_candidates']
        example['dialog_id'] = row['dialog_id']
        example['question'] = row['question'].split(SPACE)
        example['qn_entities'] = row['qn_entities']
        example['relations'] = row['relations']
        example['sources'] = row['sources']
        example['targets'] = row['targets']
        example['utterances'] = row['utterances']

        maxlen['question'] = max(len(example['question']), maxlen['question'])
        maxlen['qn_entities'] = max(len(example['qn_entities']), maxlen['qn_entities'])
        #maxlen['ans_candidates'] = max(len(example['ans_entities']), maxlen['ans_entities'])
        maxlen['sources'] = max(len(example['sources']), maxlen['sources'])
        maxlen['relations'] = maxlen['sources']
        maxlen['targets'] = maxlen['sources']
  return maxlen

def read_file_as_dict(input_path):
  d = {}
  with open(input_path) as input_file:
    reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
    for row in reader:
      d[row['col1']] = int(row['col2'])
  return d

class DatasetReader(object):
  def __init__(self, args, maxlen, share_idx=True):
    self.share_idx = share_idx
    word_idx = read_file_as_dict(args.word_idx)
    self.word_idx_size = len(word_idx)
    entity_idx = read_file_as_dict(args.entity_idx)
    self.entity_idx_size = len(entity_idx)
    relation_idx = read_file_as_dict(args.relation_idx)
    self.relation_idx_size = len(relation_idx)
    idx = read_file_as_dict(args.idx)
    self.idx_size = len(idx)
    
    with open(args.input_examples, 'r') as input_examples_file:
      reader = json.load(input_examples_file)
      self.maxlen = maxlen
      self.num_examples = 0
      examples = []
      for row in tqdm(reader):
        example={}
        example['ans'] = row['ans']
        example['ans_candidates'] = row['ans_candidates']
        example['dialog_id'] = row['dialog_id']
        example['question'] = row['question'].split(SPACE)
        example['qn_entities'] = row['qn_entities']
        example['relations'] = row['relations']
        example['sources'] = row['sources']
        example['targets'] = row['targets']
        example['utterances'] = row['utterances']
        ##
        self.num_examples += 1
        examples.append(example)
      vec_examples = []
      for example in tqdm(examples):
        vec_example = {}
        for key in example.keys():
          encoder = None
          if key == 'question':
            encoder = word_idx
          elif key == 'relations':
            encoder = relation_idx
          else:
            encoder = entity_idx
          #override the dict to be used in encoding if dict has to be shared
          if self.share_idx:
            encoder = idx
          #answers are always encoded by entity_idx !!!
#          if key == 'ans_entities':
#            encoder = entity_idx
          if key=='ans':
              vec_example[key]={}
              vec_example[key]['utterance'] = [encoder[word] for word in example[key]['utterance'].split(SPACE)]
              vec_example[key]['candidate_id'] = example[key]['candidate_id']
          elif key=='ans_candidates':
              vec_example[key]=[]
              for candidate in example[key]:
                  can_example={}
                  can_example['candidate_id']=candidate['candidate_id']
                  can_example['utterance']=[encoder[word] for word in candidate['utterance'].split(SPACE)]
                  vec_example[key].append(can_example)
          elif key=='dialog_id':
              vec_example[key]=example[key]
          elif key=='qn_entities': 
              if len(example[key])>0:
                  vec_example[key] = [encoder[word] for word in example[key]]
              else:
                  vec_example[key] = 0
          elif key=='question':
              vec_example[key] = [encoder[word] for word in example[key]]
          elif key=='relations' or key=='sources' or key=='targets':
              vec_example[key] = [encoder[word] for word in example[key]]
          elif key=='utterances':
              vec_example[key]=[]
              for utterance in example[key]:
                  vec_example[key].append([encoder[word] for word in utterance.split(SPACE)])
#          if key == 'ans_entities':
#            # answers should be in [0,count_entities-1]!!!
#            vec_example[key] = [label - 1 for label in vec_example[key]]

        vec_examples.append(vec_example)
    self.vec_examples = vec_examples

  def get_examples(self):
    return self.vec_examples

  def get_max_lengths(self):
    return self.maxlen

  def get_word_idx_size(self):
    return self.get_word_idx_size()

  def get_relation_idx_size(self):
    return self.relation_idx_size

  def get_entity_idx_size(self):
    return self.entity_idx_size

  def get_idx_size(self):
    return self.idx_size


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the kv file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  parser.add_argument('--idx', help='overall vocabulary', required=True)
  args = parser.parse_args()
  dr = DatasetReader(args,maxlen)