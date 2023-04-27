# Multimodal-Propaganda-Detection

The website of the shared task, with the submission instructions, updates on the competition and the live leaderboard can be found here: https://propaganda.math.unipd.it/neurips2023/

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

**Task 1:** Given a meme, identify whether it contains a persuasion technique. This is a multilabel classification problem.

**Task 2:** Given a meme, identify which persuasion techniques, organized in a hierarchy, are used both in the textual and in the visual content of the meme (multimodal task). If the ancestor node of a technique is selected, only partial reward wil be given. This is a hierarchical multilingual multilabel classification problem.

## Data Format

The datasets are JSON files. The text encoding is UTF-8.

### Input data format

#### Task 1:
An object of the JSON has the following format:
```
{
  id -> identifier of the example,
  label -> propaganda or non-propaganda,
  text -> textual content of meme,
  image -> name of the image file containing the meme
}
```
##### Example
```
{
        "id": "1234",
        "label": propaganda,
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
In the case of task 2, each entry of the field labels must include the fields "start", "end", "technique". We provide format checkers to automatically check the format of the submissions (see below). 

If you want to check the performance of your model on the development and test (when available) sets, upload your predictions' file to the website of the shared task: https://propaganda.math.unipd.it/neurips2023/. 
See instructions on the website about how to register and make a submission. 

## Format checkers

The format checkers for the subtasks 1 and 2 are located in the [format_checker](format_checker) module of the project. 
Each format checker verifies that your generated results file complies with the expected format. 
The format checker for subtask 2 is included in the scorer. 

Before running the format checker please install all prerequisites through,
> pip install -r requirements.txt

To launch it, please run the following command:

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
The **official evaluation metric** for the task is **macro-F1**. However, the scorer also reports macro-F1. 

To launch it, please run the following command:
```python
python3 scorer/task1_3.py --gold_file_path=<path_to_gold_labels> --pred_file_path=<path_to_your_results_file> --classes_file_path=<path_to_techniques_categories_for_task>
```

Note: You can set a flag ```-d```, to print out more detailed scores.

### Task 2:
The **official evaluation metric** for the task is **macro-F1**. However, the scorer also reports macro-F1. 

To launch it, please run the following command:
```python
python3 scorer/task1_3.py --gold_file_path=<path_to_gold_labels> --pred_file_path=<path_to_your_results_file> --classes_file_path=<path_to_techniques_categories_for_task>
```



## Baselines

### Task 1

 * Random baseline
 ```
cd baselines; python3 baseline_task1_random.py
 ```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.04494.

### Task 2

The baseline for task 2 simply creates random spans and technique names for the development set. No learning is performed. 
Run as
```
cd baselines; python3 baseline_task2.py
```
If you submit the predictions of the baseline on the development set to the shared task website, you would get a F1 score of 0.00699.
If you score the baseline on the training set (uncomment lines 5-6 in baseline_task2.py), you should get a F1 score of 0.038112
```
python3 task-2-semeval21_scorer.py -s ../../baselines/baseline-output-task2-train.txt -r ../../data/training_set_task2.txt -p ../../techniques_list_task1-2.txt 
...
F1=0.00699
...
```

## Licensing

These datasets are free for general research use.
