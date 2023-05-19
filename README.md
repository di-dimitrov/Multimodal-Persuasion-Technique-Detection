# Multimodal-Persuasive-Technique-Detection

The website of the shared task, with the submission instructions, updates on the competition and the live leaderboard can be found here: https://propaganda.math.unipd.it/neurips2023competition/

__Table of contents:__

- [Competition](#semeval-2021-task6-corpus)
  - [List of Versions](#list-of-versions)
  - [Task Description](#task-description)
  - [Data Format](#data-format)
  - [Format checkers](#format-checkers)
  - [Scorers](#scorers)
  - [Baseline](#baseline)
  - [Licensing](#licensing)
  - [Citation](#citation)

## List of Versions
* __v1.2 [XXXX/XX/XX]__ - Gold labels for dev data for tasks 1 and 2 released.
* __v1.1 [XXXX/XX/XX]__ - Development data for tasks 1 and 2 released.
* __v1.0 [XXXX/XX/XX]__ - Training data for tasks 1 and 2 released.


## Task Description

**Task 1:** Given a meme, identify whether it contains a persuasion technique or no techniques are used in the meme. This is a binary classification problem.

**Task 2:** Given a meme, identify which persuasion techniques organized in a hierarchy, are used both in the textual and in the visual content of the meme (multimodal task). If the ancestor node of a technique is predicted, only partial reward is given (see evaluation metrics for details). This is a hierarchical multilingual multilabel classification problem. 
The list of techniques, together with definitions and examples, are available [on the competition website](https://propaganda.math.unipd.it/neurips2023competition/definitions22.html). The list of ancestors of each technique is described in the file [hierarchy-rewards.txt](hierarchy-rewards.txt). 

## Data Format

The datasets are JSON files. The text encoding is UTF-8.

### Input data format

#### Task 1:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  label -> ‘propandistic’ or ‘not-propagandistic’,
  text -> textual content of meme,
  image -> name of the image file containing the meme
}
```
##### Example
```
{
        "id": "1234",
        "label": ‘propandistic’,
        "text": "I HATE TRUMP\n\nMOST TERRORIST DO",
        "image" : "prop_meme_1234.png"
}
```
#### Task 2:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  text -> textual content of meme
  image -> name of the image file containing the meme
  labels -> list of propaganda techniques appearing in the meme (based on hierarchy
}
```
##### Example
```
{
        "id": "125",
        "labels": [
            "Reductio ad hitlerum",
            "Smears",
            "Loaded Language",
            "Name calling/Labeling"
        ],
        "text": "I HATE TRUMP\n\nMOST TERRORIST DO",
        "image": "125_image.png"
}
```
<!--![125_image](https://user-images.githubusercontent.com/33981376/99262849-1c62ba80-2827-11eb-99f2-ba52aa26236a.png)-->
<img src="https://user-images.githubusercontent.com/33981376/99262849-1c62ba80-2827-11eb-99f2-ba52aa26236a.png" width="350" height="350">

### Prediction Files Format

A prediction file, for example for the development set, must be one single JSON file for all memes. The entry for each meme must include the fields "id" and "labels". As an example, the input files described above would be also valid prediction files. 
We provide format checkers to automatically check the format of the submissions (see below). 

If you want to check the performance of your model on the development and test (when available) sets, upload your predictions' file to the website of the shared task: https://propaganda.math.unipd.it/neurips2023/. 
See instructions on the website about how to register and make a submission. 

## Format checkers

The format checkers for tasks 1 and 2 are located in the [format_checker](format_checker) module of the project. 
Each format checker verifies that your generated results file complies with the expected format. 
The format checker for subtask 2 is included in the scorer. 

Before running the format checker please install all prerequisites through,
> pip install -r requirements.txt

To launch it, run the following command:

```python
python3 format_checker/task1_3.py --pred_files_path=<path_to_your_results_files> --classes_file_path=<path_to_techniques_categories_for_task>
```
Note that the checker can not verify whether the prediction file you submit contain all lines, because it does not have access to the corresponding gold file.


## Scorer and Official Evaluation Metrics

The scorer for the subtasks is located in the [scorer](scorer) module of the project.
The scorer will report official evaluation metric and other metrics of a prediction file.

You can install all prerequisites through,
> pip install -r requirements.txt

### Task 1:
The **official evaluation metric** for the task is **macro-F1**. However, the scorer also reports micro-F1. 

To launch it, please run the following command:
```python
python3 scorer/task1.py --gold_file_path=<path_to_gold_labels> --pred_file_path=<path_to_your_results_file> --classes_file_path=<path_to_techniques_categories_for_task>
```

Note: You can set a flag ```-d```, to print out more detailed scores.

### Task 2:
The **official evaluation metric** for the task is a modified version of the **micro-F1** that allows for partial matchings according to the hierarchy of techniques defined in `hierarchy-rewards.txt`. 
The leaf nodes in the hierarchy are the [22 techniques](https://propaganda.math.unipd.it/neurips2023competition/definitions22.html), while internal nodes are grouping of them, according to their characteristics. For instance, "Distraction" is a supercategory for the techniques "Straw man", "Red Herring" and "Whataboutism", since they all have the goal of distracting from the main thesis of the opponent. If an output label is Distraction while the gold label is "Red Herring", a partial reward is given. The supercategories are described [here](https://knowledge4policy.ec.europa.eu/sites/default/files/JRC132862_technical_report_annotation_guidelines_final_with_affiliations_1.pdf).    
The modified micro_F1 is computed as follows: 

$$ Prec=\frac{tpw}{tp+fp} $$

$$ Rec=\frac{tpw}{tp+fn} $$

$$ microF1=2\frac{Prec\cdot Rec}{Prec+Rec}, $$

where the standard definitions of tp (true positive), fp (false positive), fn (false negative) are modified as follows:
  - tp=1 if 
    - the prediction is a leaf-node in the hierarchy (a technique, not a supercategory) and it is correct or 
    - the prediction is an ancestor of the gold label in the hierarchy;
  - tpw is the partial reward for predicting an ancestor of the technique
  - fp = 1 if no gold label is a descendant of the predicted label
  - fn = 1 if a gold labels or its ancestors has not matched any predicted label

To ensure predictions get the highest reward, they are matched by depth in thehierarchy - first the leafs, then their parents etc...
The function avoids the same prediction label matching more than one gold label 


To launch the scorer run the following command:
```python
python3 scorer/task2.py --gold_file_path=<path_to_gold_labels> --pred_file_path=<path_to_your_results_file> -h=<path_to_the_hierarchy>
```
The set of all output labels, thus the techniques and all internal labels, are defined in `hierarchy-rewards.txt`, but they can be obtained with the previous command by adding a -t flag (notice that in this case --gold_file_path and --pred_file_path need not point to existing files):
```python
python3 scorer/task2.py --gold_file_path=<path_to_gold_labels> --pred_file_path=<path_to_your_results_file> -h=<path_to_the_hierarchy> -t
```


## Baselines

### Task 1

 * Random baseline
 ```
cd baselines; python3 baseline_task1_random.py
 ```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.04494.

### Task 2

TODO 

Run as
```
cd baselines; python3 baseline_task2.py
```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of XXX. 
```
python3 scorer/task2.py -s baselines/baseline-output-task2-train.txt -r data/training_set_task2.txt -w hierarchy-rewards.txt 
...
F1=XXX
...
```

## Licensing

The dataset is available on the [competition website](https://propaganda.math.unipd.it/neurips2023competition/). 
You'll have to sign an online agreement before downloading and using our data, which is strictly for research purposes only and cannot be redistributed or used for malicious purposes such as but not limited to manipulation, targeted harassment, hate speech, deception, and discrimination.

## Contact
You can contact us at <TODO: create email and Slack channel>
