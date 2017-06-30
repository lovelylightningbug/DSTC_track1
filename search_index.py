#!/usr/bin/python

import codecs
import unicodedata
import csv
"""
Stores the clean_wiki-entities_kb_doc.txt file as a whoosh inverted index file with entity_name, fieldname(relation), content 
as the schema. The documents are the doc contents for each entity.

First stopwords are removed from the question, then it is converted into query objects by the queryparser,
then the query objects are used to return the top 20(limit) matching documents from the inverted index.

From those matching documents the respective entities are returned.


"""
from whoosh import qparser
from whoosh import scoring
from whoosh.index import create_in
from whoosh.fields import *

from whoosh.qparser import QueryParser
"""
The job of a query parser is to convert a query string submitted by a user
into query objects (objects from the whoosh.query module).

For example, the user query:

'rendering shading'

might be parsed into query objects like this:

And([Term("content", u"rendering"), Term("content", u"shading")])
"""

from whoosh.filedb.filestore import RamStorage

def read_file_as_dict(input_path):
      d = {}
      with open(input_path) as input_file:
        reader = csv.DictReader(input_file, delimiter='\t', fieldnames=['col1', 'col2'])
        for row in reader:
          d[row['col1']] = int(row['col2'])
      return d    

class SearchIndex(object):
  def __init__(self, doc_path, stopwords=None):
    st = RamStorage()
    st.create()
    schema = Schema(entity1_name=TEXT(stored=True), fieldname=TEXT(stored=True), entity2_name=TEXT(stored=True))
    self.ix = st.create_index(schema)
    writer = self.ix.writer()
    self.remove_stopwords_while_indexing = False
    if stopwords:
      self.remove_stopwords_while_indexing = True
      self.stopwords_dict = read_file_as_dict(stopwords)

    with open(doc_path, 'r') as graph_file:
        reader = csv.DictReader(graph_file, delimiter="\t", fieldnames=['e1_relation', 'e2'])
        for row in tqdm(reader):
            entity_relation, e2 = row['e1_relation'], row['e2']
            tokens=entity_relation.split()
            e1=tokens[1]
            relation=tokens[2]
            writer.add_document(entity1_name=e1, fieldname=relation, entity2_name=e2)
    writer.commit()

  def remove_stopwords_from_text(self, content):
    words = content.split(SPACE)
    words_clean = []
    for word in words:
      if self.remove_stopwords_while_indexing and word not in self.stopwords_dict:
        words_clean.append(word)
    return " ".join(words_clean) if len(words_clean) > 0 else content

  def get_candidate_docs(self, question, limit=20):
    docs = set([])
    question = self.remove_stopwords_from_text(question)
    with self.ix.searcher() as searcher:
      query = QueryParser("content", self.ix.schema, group=qparser.OrGroup).parse(question)
      results = searcher.search(query, limit=limit)
      for result in results:
        docs.add(result['entity_name'])
    docs = [unicodedata.normalize('NFKD', doc).encode('ascii','ignore') for doc in docs]
    return docs
  
 
      
if __name__=="__main__":
  searcher = SearchIndex("../data/extendedkb1.txt","../data/stopwords.txt")
  #print searcher.get_candidate_docs("ginger rogers and")
