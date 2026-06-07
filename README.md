# Digital Twin for Fault Diagnosis

> This project was aiming to train an AI model to detect failure on a 4 motors robot. In fact, the model is train on a large simulated dataset and is use on real data. The gap between the two distribution brings a drop in the performance of the previous models.

# The Problematic 

> So, the main objective of the project was to increase the accuracy of the model on the real data. 



## ✨ Work achieved 

> Increase the number of training data
> Add some noise to have a closer distribution of the simulated data. 



## 🚀 Installation

### Prérequis

- Matlab 
- Accès to La Ruche

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
On La Ruche run the following command
```bash
sbatch Transfer-learning/lancer-calcul.sh
```



### Cas d'usage

To see the result of the training open the file confusion_matric_real_rf.png and confusion_matric_rf.png

---

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
