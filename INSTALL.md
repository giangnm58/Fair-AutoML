# Installation and Usage
To run Fair-AutoML, we need to install Python 3 environment on Linux. 
The current version has been tested on Python 3.10. 

### Environment Setup
Follow these steps to clone the Fairify repository and install Fair-AutoML.

1. Clone this repository and move to the directory:

```
git clone https://github.com/giangnm58/Fair-AutoML.git
cd Fair-AutoML/
```

1. Clone this repository and move to the directory:

```
git clone https://github.com/giangnm58/Fair-AutoML.git
cd Fair-AutoML/
``` 

2. Navigate to the cloned repository: `cd Fair-AutoML/` and install required packages:

```
pip install -r requirements.txt
```

### Run the Fair-AutoML tool
To replicate these outcomes, follow these steps: Access the evaluation section, where you'll encounter four distinct folders. These folders correspond to the evaluation source codes for four specific datasets: Adult Census, Bank Marketing, German Credit, and Titanic.

Inside each folder, you'll discover the pertinent source code. The name of each source file signifies the combination of the dataset, the machine learning algorithm employed, and the fairness metric assessed.

Contained within these source code are the necessary components: the dataset itself, buggy models, and a streamlined search space that has undergone pruning.
#### Experiment 1: Adult Census Dataset
**Example:** To improve Average Odds Difference (AOD) metric for a Gradient Boosting Classifier (GBC) on Adult Census (adult) dataset run:
```
./fairify-stress.sh <dataset>
```
