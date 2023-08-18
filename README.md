# Fair-AutoML
This repository contains the source code, benchmark models, and datasets for the paper - **"Fix Fairness, Don’t Ruin Accuracy: Performance Aware Fairness Repair using AutoML"**, appeared in ESEC/FSE 2023 at San Francisco, California, United States.

### Authors
* Giang Nguyen, Iowa State University (gnguyen@iastate.edu)
* Sumon Biswas, Carnegie Mellon University (sumonb@cs.cmu.edu)
* Hridesh Rajan, Iowa State University (hridesh@iastate.edu)

**PDF** https://arxiv.org/abs/2306.09297

Fair-AutoML is an extension of Auto-Sklearn, which is used to repair fairness bugs.

* The source code of this work is built on top Auto-Sklearn.
* To reproduce the results, run files in the "evaluation" folder. Each file contains the dataset, buggy models, and the pruned search space. The name of each file represents "dataset + ML algorithm + fairness metric."
* The dataset can be found in the "dataset" folder.

### Cite the paper as
```
@article{nguyen2023fix,
  title={Fix Fairness, Don't Ruin Accuracy: Performance Aware Fairness Repair using AutoML},
  author={Nguyen, Giang and Biswas, Sumon and Rajan, Hridesh},
  journal={arXiv preprint arXiv:2306.09297},
  year={2023}
}
```
