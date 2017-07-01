#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 00:26:44 2017

@author: suman
"""
import csv
from collections import defaultdict
from tqdm import tqdm
import json
from nltk.corpus import stopwords



words=set([])
entities=set([])
relations=set([])
all=set([])


def add_entity(entity):
  entities.add(entity)
  for word in entity.split(" "):
    words.add(word)

def get_utterances(dataset):
   list_utterances=[]
   for dialog in task1_data:
       for utterance in dialog['utterances']:
           list_utterances.append(utterance)
       for candidate in dialog['candidates']:
           list_utterances.append(candidate['utterance'])
   return list_utterances

def get_tokens(docs):
    tokens=[]
    for row in docs:
        utterance=row.split(" ")
        tokens.extend(utterance)
    return tokens
            

def read_kb_file(graph_path):
  with open(graph_path, 'r') as graph_file:
    reader = csv.DictReader(graph_file, delimiter="\t", fieldnames=['e1_relation', 'e2'])
    for row in tqdm(reader):
      entity_relation, entity2 = row['e1_relation'], row['e2']
      tokens=entity_relation.split()
      entity1=tokens[1]
      relation=tokens[2]
      add_entity(entity1)
      add_entity(entity2)
      relations.add(relation)
      relations.add("INV_"+relation)

def union(*sets):
  target_set = set([])
  for s in sets:
    target_set = target_set.union(s)
  return target_set

def write_ids(ids_path, s):
  ordered = sorted(s)
  id = 1
  with open(ids_path, 'w') as ids_file:
    writer = csv.DictWriter(ids_file, delimiter="\t", fieldnames=['x', 'count'])
    for x in ordered:
      writer.writerow({'x': x, 'count': id})
      id = id + 1


if __name__=="__main__":
   path='../data/'
   with open(path+'dialog-task1API-kb1_atmosphere-distr0.5-trn10000.json','rb') as fd:
       task1_data=json.load(fd)
   
   with open(path+'dialog-task2REFINE-kb1_atmosphere-distr0.5-trn10000.json','rb') as fd:
       task2_data=json.load(fd)
   
   with open(path+'dialog-task3OPTIONS-kb1_atmosphere-distr0.5-trn10000.json','rb') as fd:
       task3_data=json.load(fd)

   with open(path+'dialog-task4INFOS-kb1_atmosphere-distr0.5-trn10000.json') as fd:
       task4_data=json.load(fd)
   
   with open(path+'dialog-task5FULL-kb1_atmosphere-distr0.5-trn10000.json') as fd:
       task5_data=json.load(fd)
   
   total_utterance_list=get_utterances(task1_data)
   total_utterance_list.extend(get_utterances(task2_data))
   total_utterance_list.extend(get_utterances(task3_data))
   total_utterance_list.extend(get_utterances(task4_data))
   total_utterance_list.extend(get_utterances(task5_data))
   
   tokens=get_tokens(total_utterance_list)
   for word in tokens:
       words.add(word)
   
   kb_path=path+'extendedkb1.txt'
   read_kb_file(kb_path)
   write_ids(path+'entity_ids.txt',entities)
   write_ids(path+'relation_ids.txt',relations)
   write_ids(path+'word_ids.txt',words)
   stop=set(stopwords.words('english'))
   write_ids(path+'stopwords.txt',stop)
   
   all = union(words, entities, relations)
   write_ids(path+'ids.txt',all)
   
   