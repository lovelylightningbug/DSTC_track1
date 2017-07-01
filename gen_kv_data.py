#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 00:02:42 2017

@author: suman
"""

import argparse
import csv
import random
import json

from knowledge_graph import KnowledgeGraph
from question_parser import QuestionParser

from tqdm import tqdm

MAX_RELEVANT_ENTITIES = 4
HOPS_FROM_QN_ENTITY = 1
MAX_CANDIDATE_ENTITIES = 1024
MAX_CANDIDATE_TUPLES = 2048


def read_file_as_dict(input_path):
  d = {}
  with open(input_path) as input_file:
    reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
    for row in reader:
      d[row['col1']] = int(row['col2'])
  return d

def union(*sets):
  target_set = set([])
  for s in sets:
    target_set = target_set.union(s)
  return target_set

def get_neighboring_entities(entities, num_hops=2):
  nbr_entities = set([])
  for entity in entities:
    for nbr in knowledge_base.get_candidate_neighbors(entity, num_hops=num_hops,
                                                      avoid_high_degree_nodes=True):
      nbr_entities.add(nbr)
  return nbr_entities

def get_tuples_involving_entities(candidate_entities):
  tuples = set([])
  for s in candidate_entities:
    if s in knowledge_base.get_high_degree_entities():
      continue
    for t in knowledge_base.get_adjacent_entities(s):
      r = knowledge_base.get_relation(s,t)
      tuples.add((s, r, t))
  return tuples

def get_str_of_seq(entities):
  return "|".join(entities)

def extract_dimension_from_tuples_as_list(list_of_tuples, dim):
  result = []
  for tuple in list_of_tuples:
    result.append(tuple[dim])
  return result

def main(args):
    with open(args.input_examples, 'r') as input_examples_file:
#        with open(args.output_examples, 'w') as output_examples_file:
            reader = json.load(input_examples_file)
            #reader = csv.DictReader(input_examples_file, delimiter='\t', fieldnames=['question', 'answer'])
#            writer = csv.DictWriter(output_examples_file, delimiter='\t',
#                                      fieldnames=['question', 'qn_entities', 'ans_entities',
#                                               'sources', 'relations', 'targets'])
            lst=[]
            for story in tqdm(reader):
                dialog_id=story['dialog_id']
                ans=story['answer']
                ans_candidates=story['candidates']
                question=story['utterances'][-1]
                qn_entities=question_parser.get_question_entities(question)
                utterances=story['utterances']
                relevant_entities=question_parser.get_question_entities(" ".join(utterances))
                nbr_qn_entities=get_neighboring_entities(relevant_entities, num_hops=HOPS_FROM_QN_ENTITY)
                candidate_entities=union(qn_entities,relevant_entities,nbr_qn_entities)
                if len(candidate_entities) > MAX_CANDIDATE_ENTITIES:
                    candidate_entities = set(random.sample(candidate_entities, MAX_CANDIDATE_ENTITIES))
                tuples = get_tuples_involving_entities(candidate_entities)
                if len(tuples) > MAX_CANDIDATE_TUPLES:
                    tuples = set(random.sample(tuples, MAX_CANDIDATE_TUPLES))
                sources = extract_dimension_from_tuples_as_list(tuples, 0)
                relations = extract_dimension_from_tuples_as_list(tuples, 1)
                targets = extract_dimension_from_tuples_as_list(tuples, 2)
                output_row = {
                  'dialog_id':dialog_id,
                  'question': question,
                  'qn_entities': list(qn_entities),
                  'ans': ans,
                  'ans_candidates':ans_candidates,
                  'utterances':utterances,
                  'sources': list(sources),
                  'relations': list(relations),
                  'targets': list(targets)}
                lst.append(output_row)
            fd_out=open(args.output_examples, 'wb')
            json.dump(lst, fd_out)
            fd_out.close()

                
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--input_examples', help='the raw qa pairs', required=True)
  parser.add_argument('--input_graph', help='the graph file', required=True)
  parser.add_argument('--stopwords', help='stopwords file', required=False)
  parser.add_argument('--output_examples', help='the processed output file', required=True)
  args = parser.parse_args()

  #global variables
  knowledge_base = KnowledgeGraph(args.input_graph, unidirectional=False)
  stop_vocab = read_file_as_dict(args.stopwords)
  question_parser = QuestionParser(knowledge_base.get_entities(), stop_vocab)
  main(args)
  