# Digital Twin for Fault Diagnosis

> This project was aiming to train an AI model to detect failure on a 4 motors robot. In fact, the model is train on a large simulated dataset and is use on real data. The gap between the two distribution brings a drop in the performance of the previous models.

# The Problematic 

> So, the main objective of the project was to increase the accuracy of the model on the real data. 



## ✨ Work achieved 

> Increase the number of training data
> Add some noise to have a closer distribution of the simulated data.
> First version of transfer training
> An ameliorated version of transfer learning with corss validation



## 🚀 Installation

### Prérequis

- Matlab 
- Access to La Ruche

## Lancer un environnement sur La Ruche

```bash
module load anaconda3/2023.09-0/none-none
conda create -n my_env
source activate my _env
```

### Installer les dépendances

```bash
pip install -r requirements.txt
```

### Cloner le dépôt

```bash
git clone https://github.com/Alec129/Pole-projet/
cd Pole-projet
```


### Lancer le projet
The repository includes a submission script to run the cross-validation job on La Ruche:
From the repository root on La Ruche, submit the job with:

- First Version of transfer learning
```bash
sbatch Transfer-learning/lancer-calcul.sh
```
To see the result of the training open the file confusion_matric_real_rf.png and confusion_matric_rf.png

- Last version of transfer learning with cross validation
```bash
sbatch scripts/submit_cv5.sh
```
Example trained checkpoints are in scripts/ (e.g. best_finetuned_cv.pt, best_pretrained_cv.pt)

### Performance 
- The transfer learning model achieves an accuracy of 83.33% ± 7.86%.
- Confusion matrix (5-fold CV):

## 📁 Structure du projet

```text
project/
├── Transfer-learning/
│   ├── mydataset/
|   |   ├── Healthy/
|   |   ├── Motor_1_Stuck/
|   |   ├── Motor_1_Steady_state_error/
|   |   ├── Motor_2_Stuck/
|   |   ├── Motor_2_Steady_state_error/
|   |   ├── Motor_3_Stuck/
|   |   ├── Motor_3_Steady_state_error/
|   |   ├── Motor_4_Stuck/
|   |   ├── Motor_4_Steady_state_error/
│   ├── lancer-calcul.sh/
│   ├── transfer-learning-vf.py/
│   ├── models.py/
├── requirements.txt/
└── README.md
```

### Description des dossiers

| Dossier/Fichier | Description |
|----------|------------|
| src/Transfer-learning | All the code to run the transfer learning method|
| src/Transfer-learning/mydataset | Contain all the data to train the model |
| src/Transfer-learning/lancer-calcul.sh | file to start the calcul on La Ruche |
| src/Transfer-learning/transfer-learning.py | code of the transfer learning method |
| src/Transfer-learning/models.py | file of the model of the CNN 1D and LSTM |




## 👤 Auteur

**Alec BEOLET**



---

## 🙏 Remerciements

- Laboratoire génie industriel de CentraleSupélec
