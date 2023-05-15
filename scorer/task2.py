import json
import logging.handlers
import argparse
import os
import sys
sys.path.append('.')
import hierarchy 
"""
Scorer for shared task on Multimodal Persuasion Techniques Detection, 
task 2: "Given a meme, identify which persuasion techniques, organized in a hierarchy, are used both in the textual and in the visual content of the meme (multimodal task). If the ancestor node of a technique is selected, only partial reward wil be given. This is a hierarchical multilingual multilabel classification problem."

"""

logger = logging.getLogger("task2_scorer")
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.setLevel(logging.INFO)
#logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def read_gold_or_pred_file(file_path):
  """
  Read either the gold or the prediction file
  :param file_path: a json file with predictions or gold labels, 
    [
      {
        id     -> identifier of the example,
        labels -> a list of technique names (double quoted strings) 
      }
    ]
  The function does not perform any checks on the data
  :return: {id:label_list} dict;
  """

  labels = {}
  with open(file_path, encoding='utf-8') as f:
    labels_obj = json.load(f)
    for obj in labels_obj:
      labels[obj['id']] = obj['labels']
  return labels


def evaluate(pred_labels, gold_labels, hierarchy: hierarchy.Hierarchy):
  """
    Evaluates the predicted classes w.r.t. gold labels, returning micro_f1. 
    micro_f1 is computed as follows: 
    Prec=tpw/(tp+fp), Rec=tpw/(tp+fn), micro_f1=2*Prec*Rec/(Prec+Rec),
    where 
      tp = 1 predicting the correct technique (leaf node) or any ancestor in the hierarchy is a true positive
      tpw is the partial reward for predicting an ancestor of the technique
      fp = 1 if no gold label is a descendant of the predicted label
      fn = 1 if a gold labels or its ancestors has not matched any predicted label
    To ensure predictions get the highest reward, they are matched by depth in the hierarchy - first the leafs, then their parents etc...
    The function avoids the same prediction label matching more than one gold label 
    
    :param pred_labels: {id:label_list} dict with predicted labels, 
      as returned by read_gold_or_pred_file() 
    :param gold_labels: {id:label_list} dict with gold labels, 
      as returned by read_gold_or_pred_file().
    :param HIERARCHY: an object with the hierarchy of techniques and the 
      partial rewards for prediction an ancestor instead of the actual technique (see class Hierarchy in hierarchy.py) 
    :return: micro_f1 computed as above  
  """
  tp, fp, fn, tpw = (0, 0, 0, 0.0) 
  for example_id in pred_labels.keys():
    temp_pred = pred_labels[example_id].copy()
    for glabel in gold_labels[example_id]:
      found = False
      for i, label in enumerate(hierarchy.get_ordered_ancestors_list(glabel)):
        if label in temp_pred:
          tp+=1
          tpw+=hierarchy.get_ancestor_weight(glabel, i)
          temp_pred.remove(label) #each predicted label matches only one gold label
          found = True
          break
      if not found:
        fn+=1
    fp += len(temp_pred)
  if (tp+fp==0 and tp+fn==0): # all the examples have no technique and 
                              # no technique is ever predicted
    micro_f1 = 1.0
  else:
    precision = tpw/(tp+fp)
    recall = tpw/(tp+fn)
    micro_f1 = 2*precision*recall/(precision+recall)
  return micro_f1


def validate_prediction_and_gold_labels(pred_labels, gold_labels, h: hierarchy.Hierarchy):
  """
    checks if
      1) the ids in pred_labels and gold_labels are identical
      2) the set of techniques in each file is part of the hierarchy
      3) each gold_labels techniques is a leaf node in the hierarchy
  """
  if set(gold_labels.keys()) != set(pred_labels.keys()):
    logger.error('There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.\nGold ids:\n{}\n\nPrediction ids:\n{}'.format(sorted(gold_labels.keys()), sorted(pred_labels.keys())))
    raise ValueError('There are either missing or added examples to the prediction file. Make sure you only have the gold examples in the prediction file.\nGold ids:\n{}\n\nPrediction ids:\n{}'.format(sorted(gold_labels.keys()), sorted(pred_labels.keys())))

  for ex_labels in pred_labels.values():
    for l in ex_labels:
      if l not in h.get_unique_node_list():
        logger.error('Unrecognised technique in prediction labels: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_unique_node_list()))))
        raise ValueError('Unrecognised technique in prediction labels: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_unique_node_list()))))
  for ex_labels in gold_labels.values():
    for l in ex_labels:
      if l not in h.get_unique_node_list():
        logger.error('Unrecognised technique in gold labels: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_unique_node_list()))))
        raise ValueError('Unrecognised technique in gold labels: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_unique_node_list()))))
      if l not in h.get_leaf_nodes():
        logger.error('The gold file has a non-leaf node as technique: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_leaf_nodes()))))
        raise ValueError('The gold file has a non-leaf node as technique: {}.\nList of techniques: {}'.format(l, "\n".join(sorted(h.get_leaf_nodes()))))

  return True


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--gold_file_path",
    '-g',
    type=str,
    required=True,
    help="Paths to the file with gold annotations."
  )
  parser.add_argument(
    "--pred_file_path",
    '-p',
    type=str,
    required=True,
    help="Path to the file with predictions"
  )
  parser.add_argument(
    "--log_to_file",
    "-l",
    action='store_true',
    default=False,
    help="Set this flag if you want to log the execution to file. The log will be appended to <pred_file>.log"
  )
  parser.add_argument(
    "--hierarchy_file_path", 
    "-w", 
    required=True, 
    help="The absolute path to the file containing the hierarchy."
  )
  parser.add_argument(
    "--print_technique_list", 
    "-t", 
    required=False,
    action='store_true',
    default=False,
    help="Print the list of possible output technique names and exits."
  )
  args = parser.parse_args()

  pred_file = args.pred_file_path
  gold_file = args.gold_file_path

  if args.log_to_file:
    output_log_file = pred_file + ".log"
    logger.info("Logging execution to file " + output_log_file)
    fileLogger = logging.FileHandler(output_log_file)
    fileLogger.setLevel(logging.DEBUG)
    fileLogger.setFormatter(formatter)
    logger.addHandler(fileLogger)
    logger.setLevel(logging.DEBUG) #

  if not os.path.exists(args.hierarchy_file_path):
    logger.errors("File doesnt exists: {}".format(args.hierarchy_file_path))
    raise ValueError("File doesnt exists: {}".format(args.hierarchy_file_path))
  h = hierarchy.Hierarchy(args.hierarchy_file_path)

  if args.print_technique_list:
    logger.info("List of techniques:\n" + "\n".join(sorted(h.get_unique_node_list())))
    sys.exit()

  if args.log_to_file:
    logger.info('Reading gold file')
  else:
    logger.info("Reading gold predictions from file {}".format(args.gold_file_path))
  gold_labels = read_gold_or_pred_file(args.gold_file_path)
  if args.log_to_file:
    logger.info('Reading predictions file')
  else:
    logger.info('Reading predictions file {}'.format(args.pred_file_path))
  pred_labels = read_gold_or_pred_file(args.pred_file_path)

  if validate_prediction_and_gold_labels(pred_labels, gold_labels, h):
    logger.info('Prediction file format is correct')
    micro_f1 = evaluate(pred_labels, gold_labels, h)
    logger.info("micro-F1={:.5f}".format(micro_f1))
    if args.log_to_file:
      print("{}".format(micro_f1))
