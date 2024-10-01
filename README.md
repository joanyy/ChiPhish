This repository describes the implementation of "Beyond the West: Revealing and Bridging the Gap between Western and Chinese Phishing Website Detection" currently under review.

# **Organization**

- ```data```: containing the dataset we used in our paper;
- ```preprocessing```: including the code of our feature extractor;
- ```custom_pwd```: containing the source code of the custom ML-PWD we built;
- ```production_pwd```: containing the experiments on Virustotal and GSB;
- ```spacephish_pwd```: containing the implementation of vanilla PWD of SpacePhish.

In the root folder of this repository, we provide a ```requirements.txt``` file to clarify what Python libraries were used in our experiments. 

# **Contents**
We explain the documents in the order of the list above, i.e., ```data```, ```preprocessing```, ```custom_pwd```, ```production_pwd``` and ```spacephish_pwd```. 

## data
This folder includes 5 subfolders corresponding to 5 datasets:
- _JP_. This folder includes 100 Japanese websites: phishing websites collected from openphish, phishtank and [phishIntention](https://github.com/lindsey98/PhishIntention); benign websites are most common Japanese websites according to [similarweb](https://www.similarweb.com/).
- _KR_. This folder includes 100 Korean websites, collected from the same source with _JP_.
- _EngPhish_. This English-only data corpus is extracted from the latest public dataset [LNU-Phish](https://lnu-phish.github.io/). This folder includes raw data of webpages, which will be deleted later to avoid potential copyright violations. However, we will maintain the preprocessed version of each webpage. 
- _WstPhish_. We extracted a subset from [Zenodo](https://dl.acm.org/doi/abs/10.1145/3465481.3470112) by cosidering the most common European phonologic languages; an example is provided in the subfolder 'wstphish'. 
- _ChiPhish_. We collected a dataset for Chinese-based PWD, **which we will publicly release** upon publication. We collected Chinese benign webpages from the top60 Chinese websites listed on 'chinaz.com'; this repository includes a snippet of our full dataset. We provide the top30's homepage information (the source of benign webpages) in subfolder 'ch_benign_homepage_top30'. The preliminary information about our dataset is described in 'chphish_20samples' containing 10 benign samples and 10 phish samples. The brands distribution of ChiPhish dataset is shown below:

| Category | #Benign | Example | 
|:---------:|:-----:|:------:| 
| e-commerce | 173 | 1688.com | 
| finance  |59 |  alipay.com  | 
| education | 121 | huatu.com |  
| government | 2 | beijing.com | 
| health | 23 |haodf.com  | 
| email| 10 |163.com | 
| information & service | 267 | 58.com |
| news| 96  | ifeng.com | 
| Search engine | 22 |baidu.com| 
| Social network |35 | weixin.qq.com |  
| entertainment | 247 | iqiyi.com | 
| other | 0 | n/a | 



## preprocessing
This folder includes two files (i.e., Jupyter Notebooks):
- _LaSeTo.ipynb_: identifying the webpage language.
- _extractor.ipynb_: extracting features (new features, chinese-specific features and feature set of spacephish) from each single sample.
  
## custom_pwd
This folder contains 6 files (i.e., Jupyter Notebooks):
- _ChiPhish_PWD_custom.ipynb_: containing the custom ML_PWD training on ChiPhish dataset, and the performance on different datasets. 
- _EngPhish_PWD_custom.ipynb_: containing the custom ML_PWD training on EngPhish, and the performance on different datasets.
- _WstPhish_PWD_custom.ipynb_: containing the custom ML_PWD training on WstPhish, and the performance on different datasets.
- _Universal_PWD.ipynb_: including the code of 'universial' PWD trained on three datasets.
- _Img_Pwd_EngPhish.ipynb_: building the DL-based PWD (binary classifier) to analyze the screenshot of a webpage.
- _pwd_laseto.ipynb_: containing the custom RF_PWD exploited LaSeTo.
  
## production_pwd
This folder includes one file: _Virustotal_GSB.ipynb_, which contains the code on how we use Google Safe Browsin (GSB) and Virustotal predict the samples.

## spacephish_pwd
This folder contains 5 files and 1 folder: 

- _ChiPhish_spacephishF.ipynb_: containing the experiments of spacephish ML-PWD analyse ChiPhish.
- _EngPhish_spacephishF.ipynb_: containing the experiments of spacephish ML-PWD analyse EngPhish.
- _utils.py_: containing some custom-defined functions for developing ML models.
- _Wstphish_spacephishF.ipynb_: containing the experiments of spacephish ML-PWD analyse WstPhish.
- _Spacephish_detector.ipynb_: containing the experiments of original spacephish ML-PWD predicting WstPhish.
- _SpacePhish_par_: parameters of spacephish PWD.


# Instructions
1. Get the data and install the requirements. We recommend creating an virtual environment by Anaconda, and execute ```pip install -r requirements.txt```.
2. Extracting features by the script of *preprocessing/extractor.ipynb*
3. PWD performance. Input features to the *.ipynb* scripts of ```custom_pwd``` or ```production_pwd``` or ```spacephish_pwd``` to get the corresponding result on each dataset.  
