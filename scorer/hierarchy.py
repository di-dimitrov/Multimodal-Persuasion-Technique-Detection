import logging.handlers
import json

"""
  Common functions for scorers and format checkers of the shared task "Multimodal Persuasive Technique Detection" 
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
KEYS = ['id', 'labels']


class Hierarchy:
  """
  json file format: 
  """
  WM = None
  nodes = None
  file_path = None


  def __init__(self, file_path=None):

    if file_path is not None:
      self.file_path = file_path
      self.load_hierarchy()


  def extract_unique_nodes(self):

    if self.WM is not None:
        self.nodes = set()
        for x in self.WM.keys():
          self.nodes = self.nodes.union(set(self.WM[x]["path-to-root"]))


  def get_ancestor_weight(self, leaf_node_label, i):

    if self.WM is not None:
      if i<len(self.WM[leaf_node_label]["nodes-reward"]):
        return self.WM[leaf_node_label]["nodes-reward"][i]


  def get_node_ancestor(self, leaf_node_label, i):

    if self.WM is not None:
      if i<len(self.WM[leaf_node_label]["path-to-root"]):
        return self.WM[leaf_node_label]["path-to-root"][i]


  def get_ordered_ancestors_list(self, leaf_node_label):

    if self.WM is not None:
      if leaf_node_label not in self.get_leaf_nodes():
        print("ERROR: paths to root are only available for leaf nodes")
        return None
      return self.WM[leaf_node_label]["path-to-root"]
    

  def get_leaf_nodes(self):

    if self.WM is not None:
      return self.WM.keys()


  def get_unique_node_list(self):

    if self.WM is not None:
      return self.nodes
    

  def print_node_list(self):
    for i in self.nodes:
      print(i)


  def load_hierarchy(self):
    """
      Load the error weights. When predicting class i while the gold class is j, the reward is the weight (i,j) of this matrix: 1 means fully correct, 0 completely wrong. 
      The number and order of entries are supposed to be the one in the array CLASSES (usually loaded from the file label_list.txt): each line of the file has len(CLASSES) float numbers space separated. 
      ---
      :param file_path: file with the weights 
      :param CLASSES: list of valid output labels
      :return: {"class1__class2":weight} dict # reward for predicting class1 when the gold label is class2
    """
    with open(self.file_path, encoding='utf-8') as f:
      self.WM = json.load(f)
    self.extract_unique_nodes()
