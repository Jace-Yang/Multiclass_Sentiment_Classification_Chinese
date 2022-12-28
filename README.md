# Multiclass_Sentiment_Classification_Chinese
Final Project of Columbia COMS 4995 Applied Deep Learning

## Quick Demo

```
python3
>>> from demo.final_classifier import DeployedClassifier
>>> clf = DeployedClassifier()
>>> clf.pred('阿根廷终于赢了世界杯')
([('fear', 0.0024164714850485325), ('neutral', 0.0563773512840271), ('sad', 0.037695735692977905), ('surprise', 0.05237787961959839), ('angry', 0.020596427842974663), ('happy', 0.8305360674858093)], 'happy')
```
## Directory Introduction

- `data` contains original input

- `classifier_utils.py` and `data_utils.py` is called for several `ipynb files` to train & evaluate & deploy the models.

- `result`, `output`, and `docs` stores model output and logs, visualization code and final result

- `demo` contains a simple web app using this model

## Re-producing instructions

1. Obtain dataset: Downloaded from drive links released at [SMP2020-EWECT](`https://smp2020ewect.github.io/`)

1. Analyze the dataset: run `0_EDA.ipynb`

1. Train and cache the models: run `1_Training.ipynb`

    - It will cache the training logs into `result/training/`, which is uploaded to this repository

    - It will save the best models in `result/model/`, which is ignored when upload this repository. But you can download it directly from this [Google Drive folder](https://drive.google.com/drive/folders/1impSyTM0-kXY9bRby7BUYV2kqbnwnlEB?usp=share_link).

1. Evaluate the models and obtain stacked model: run `2_Evaluating.ipynb`

1. Run the Web application: open terminal and run `python3 3_run_demo.py`