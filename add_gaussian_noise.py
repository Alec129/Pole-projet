"""
add_gaussian_noise.py
=====================

Ajoute un bruit gaussien aux trajectoires SIMULEES (fichier MATLAB .mat) pour
rapprocher les donnees de simulation des conditions reelles du robot.

Projet : Digital Twin Robot for Fault Diagnosis - Groupe 8
Etape  : amelioration du dataset par ajout d'un bruit gaussien (sigma = 0.01).

Le fichier .mat contient des variables nommees. Le script bruite les MATRICES
(trajectoires, de forme N x 1000) et laisse intacts les VECTEURS / scalaires
(les labels, de forme N x 1). Il affiche pour chaque variable ce qu'il fait :
verifie cet affichage avant de committer.

Utilisation :
    python add_gaussian_noise.py my_dataset_test.mat my_dataset_noisy.mat
    python add_gaussian_noise.py in.mat out.mat --sigma 0.01 --seed 42
    python add_gaussian_noise.py in.mat out.mat --vars X        # bruite que X
    python add_gaussian_noise.py in.mat out.mat --skip-vars y   # ne touche pas a y

Remarque : le bruit etant aleatoire, fixe --seed pour obtenir une sortie
reproductible d'une execution a l'autre.
"""

import argparse
import os
import sys

import numpy as np
from scipy import io as sio


def main():
    parser = argparse.ArgumentParser(
        description="Ajoute un bruit gaussien aux trajectoires simulees (.mat)."
    )
    parser.add_argument("input", help="Fichier .mat d'entree (dataset simule).")
    parser.add_argument("output", help="Fichier .mat de sortie (dataset bruite).")
    parser.add_argument("--sigma", type=float, default=0.01,
                        help="Ecart-type du bruit gaussien (defaut : 0.01).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Graine aleatoire pour reproduire la meme sortie.")
    parser.add_argument("--vars", nargs="*", default=None,
                        help="Bruite UNIQUEMENT ces variables.")
    parser.add_argument("--skip-vars", nargs="*", default=[],
                        help="Ne bruite jamais ces variables (ex : labels).")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"Fichier introuvable : {args.input}")

    rng = np.random.default_rng(args.seed)
    skip_vars = set(args.skip_vars)

    try:
        mat = sio.loadmat(args.input)
    except NotImplementedError:
        sys.exit(
            "Ce .mat est au format v7.3 (HDF5), non lisible par scipy.\n"
            "Re-sauvegarde-le en MATLAB avec :  save('fichier.mat','-v7')\n"
            "ou installe 'mat73' :  pip install mat73"
        )

    out = {}
    for key, val in mat.items():
        if key.startswith("__"):
            continue  # metadonnees scipy, regenerees automatiquement a la sauvegarde

        arr = np.asarray(val)
        is_numeric = np.issubdtype(arr.dtype, np.number)
        # Une matrice de trajectoires a >= 2 dimensions ; un vecteur de labels
        # (N x 1) s'aplatit en 1 dimension -> on ne le bruite pas.
        is_matrix = np.squeeze(arr).ndim >= 2

        if args.vars is not None:
            noise_it = key in args.vars
        elif key in skip_vars:
            noise_it = False
        else:
            noise_it = is_numeric and is_matrix

        if noise_it and is_numeric:
            noise = rng.normal(0.0, args.sigma, size=arr.shape)
            out[key] = arr.astype(np.float64) + noise
            print(f"[mat] '{key}' {arr.shape} -> BRUITE")
        else:
            out[key] = val
            if not is_numeric:
                raison = "non numerique"
            elif not is_matrix:
                raison = "vecteur/scalaire (label probable)"
            else:
                raison = "exclu manuellement"
            print(f"[mat] '{key}' {arr.shape} -> inchange ({raison})")

    sio.savemat(args.output, out)
    print(f"\nFichier sauvegarde -> {args.output}  (sigma = {args.sigma})")


if __name__ == "__main__":
    main()
