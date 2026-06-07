#!/bin/bash
#SBATCH --job-name=mon_projet         # Nom du job [cite: 1954]
#SBATCH --output=%x.o%j               # Fichier de sortie (logs) [cite: 1956]
#SBATCH --time=01:00:00               # Temps max (HH:MM:SS) [cite: 1946]
#SBATCH --partition=cpu_short         # File d'attente courte [cite: 1949]
#SBATCH --ntasks=1                    # 1 seul processus [cite: 1931]

#!/bin/bash
#SBATCH --job-name=mon_projet

# On demande 1 seul nœud (machine)
#SBATCH --nodes=1
# On demande 4 cœurs sur cette machine (ce qui sera lu par le Canvas)
#SBATCH --cpus-per-task=40
# On demande la mémoire vive totale (ex: 16 Go)
#SBATCH --mem=16G

echo "START"

source ~/.bashrc
echo "BASHRC OK"

source activate env_stable
echo "CONDA OK"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
echo $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION

which python
python --version

echo "PYTHON START"

python /gpfs/users/beoletal/Transfer-learning/transfer-learning-real-data.py

echo "END"
