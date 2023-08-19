# Installation and Usage
To run Fair-AutoML, we need to install Python 3 environment on Linux. 
The current version has been tested on Python 3.10. 

### Environment Setup
Follow these steps to clone the Fair-AutoML repository and install Fair-AutoML.

1. Clone this repository and move to the directory:

```
git clone https://github.com/giangnm58/Fair-AutoML.git
cd Fair-AutoML/
``` 

2. Navigate to the cloned repository: `cd Fair-AutoML/` and install required packages:

```
pip3 install -r requirements.txt
```

### Run the Fair-AutoML tool
To replicate Fair-AutoML's outcomes, follow these steps: Access the evaluation section, where you'll encounter four distinct folders. These folders correspond to the evaluation source codes for four specific datasets: Adult Census, Bank Marketing, German Credit, and Titanic.

Inside each folder, you'll discover the pertinent source code. The name of each source file signifies the combination of the dataset, the machine learning algorithm employed, and the fairness metric assessed.

Contained within these source code are the necessary components: the dataset itself, buggy models, and a streamlined search space that has undergone pruning.
#### Experiment 1: Adult Census Dataset
**Example:** To enhance the Average Absolute Odds Difference (AOD) metric for a buggy Gradient Boosting Classifier (GBC) model applied to the Adult Census (adult) dataset, execute the following command:
```
cd evaluation/adult/
python3 adult_GBC_AOD.py
```

#### Experiment 2: Bank Marketing Dataset
**Example:** To enhance the Disparate Impact (DI) metric for a buggy Random Forest (RF) model applied to the Bank Marketing (bank) dataset, execute the following command:
```
cd evaluation/bank/
python3 bank_RF_DI.py
```

#### Experiment 3: German Credit Dataset
**Example:** To enhance the Equal Opportunity Difference (EOD) metric for a buggy K-Nearest Neighbors (KNN) model applied to the German Credit (german) dataset, execute the following command:
```
cd evaluation/german/
python3 german_KNN_EOD.py
```

#### Experiment 4: Titanic Dataset
**Example:** To enhance the Statistical Parity Difference (SPD) metric for a XGBoost (XGB) model applied to the Titanic (titanic) dataset, execute the following command:
```
cd evaluation/titanic/
python3 titanic_XGB_SPD.py
```

