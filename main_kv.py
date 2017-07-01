#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:36:24 2017

@author: suman
"""

import argparse
import os
import numpy as np
import random
import tensorflow as tf
import tqdm

from kv_dataset_reader import DatasetReader
from kv_dataset_reader import get_maxlen
from model_kv import KeyValueMemNN


flags = tf.app.flags
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
flags.DEFINE_integer("evaluation_interval",5, "Evaluate and print results every x epochs")
flags.DEFINE_integer("batch_size", 100, "Batch size for training.")
flags.DEFINE_integer("hops", 2, "Number of hops in the Memory Network.")
flags.DEFINE_integer("epochs", 30, "Number of epochs to train for.")
flags.DEFINE_integer("embedding_size", 128, "Embedding size for embedding matrices.")
flags.DEFINE_integer("dropout_memory", 1.0, "keep probability for keeping a memory slot")
flags.DEFINE_string("checkpoint_dir", 'checkpoints', "checkpoint directory [checkpoints]")
flags.DEFINE_integer("max_slots", 64, "maximum slots in the memory")


FLAGS = flags.FLAGS
QUESTION = "question"
QN_ENTITIES = "qn_entities"
ANS_ENTITIES = "ans_entities"
SOURCES = "sources"
RELATIONS = "relations"
TARGETS = "targets"
ANSWER = "answer"
KEYS = "keys"
VALUES = "values"



def main(args):
    max_slots=FLAGS.max_slots
    maxlen = get_maxlen(args.train_examples, args.test_examples, args.dev_examples)
    maxlen[KEYS], maxlen[VALUES] = min(maxlen[SOURCES], max_slots), min(maxlen[SOURCES], max_slots)
    args.input_examples = args.train_examples
    train_reader = DatasetReader(args, maxlen, share_idx=True)
    train_examples = train_reader.get_examples()
    args.input_examples = args.test_examples
    test_reader = DatasetReader(args, maxlen, share_idx=True)
    test_examples = test_reader.get_examples()
    args.input_examples = args.dev_examples
    dev_reader = DatasetReader(args, maxlen, share_idx=True)
    dev_examples = dev_reader.get_examples()
    batch_size = FLAGS.batch_size
    num_train = len(train_examples)
    batches = zip(range(0, num_train - batch_size, batch_size), range(batch_size, num_train, batch_size))
    batches = [(start, end) for start, end in batches]
    with tf.Session() as sess:
        model = KeyValueMemNN(sess, maxlen, train_reader.get_idx_size(), train_reader.get_entity_idx_size())
        if os.path.exists(os.path.join(FLAGS.checkpoint_dir, "model_kv.ckpt")):
          saver = tf.train.Saver()
          save_path = os.path.join(FLAGS.checkpoint_dir, "model_kv.ckpt")
          saver.restore(sess, save_path)
          print("Model restored from file: %s" % save_path)
    


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Specify arguments')
  parser.add_argument('--train_examples', help='the train file', required=True)
  parser.add_argument('--test_examples', help='the test file', required=True)
  parser.add_argument('--dev_examples', help='the dev file', required=True)
  parser.add_argument('--word_idx', help='word vocabulary', required=True)
  parser.add_argument('--entity_idx', help='entity vocabulary', required=True)
  parser.add_argument('--relation_idx', help='relation vocabulary', required=True)
  parser.add_argument('--idx', help='overall vocabulary', required=True)
  args = parser.parse_args()
  main(args)