# Monomer Featurizer and Predictor of Polymer Properties

This script provides physically grounded and interpretable featurization
of monomers for Quantitative Structure–Property Relationship modeling.
It is designed for applications in polymer informatics and materials science.

Includes GPR models for prediction of polymer properties: density, glass transition temperature, thermal decomposition temperature, melting temperature.

Citation request:
------------------
If you use this code or models in your research, please cite the following paper:

Gleb Averochkin, Ivan Zlobin, Ivan Bespalov, Eugeniy Alexandrov, Machine Learning with Physically Grounded, Interpretable Descriptors for Polymer Property Prediction and Monomer Design, Chemical Engineering Science

Features:
---------
- Classification of atoms into groups by predominant intermolecular interaction (van der Waals, dipole-dipole, ionic etc.), main/side chain, chemical element.
- Calculation of  van der Waals volumes of atomic groups (Å³).
- SMARTS-based functional group recognition.
- Calculation of numbers of functional groups and structural motives in the main and side chains.
- Computation of polymerization site distance (Å).
- Computation of chemical bond statistics.
- Integration of quantum-chemical descriptors (HOMO/LUMO, dipole moment, Mulliken charges) from xTB package.
- Computation of selected interpretable RDKit descriptors.

Dependencies:
-------------
- RDKit
- NumPy
- SciPy 
- pandas
- joblib
- xTB
- autogpr 

Usage:
------
Import environment.linux.yml and requirements.linux.txt files to set up the linux environment. For windows environment import environment.windows.yml and requirements.windows.txt files

Import monomer_featurizer_predictor.py to generate descriptors from monomer SMILES. SMILES must have two polymerization sites, marked by * symbols.

The descriptors can be used as input features for machine learning models to predict polymer properties.

The monomer_featurizer_predictor.py file also contains functions for predicting the properties of corresponding polymers. To do it, you need to import the GPR models in the same directory.

Installation:
-----------
This project relies on RDKit and xTB for molecular and quantum-chemical descriptor calculation and is developed and tested on Linux.

Linux(recommended)

1) Create the Conda environment:

```python
conda env create -f environment.linux.yml --name gleb_env
conda activate gleb_env
```

2) Install Python dependencies:

```python
python -m pip install -r requirements.linux.txt
python -m pip install git+https://github.com/IvanBespalov64/autogpr.git
```

This is the officially supported and fully reproducible setup, matching the server environment used in this project.

Windows (recommended via WSL2)
-----------------------------
Native Windows installations may lead to inconsistent behavior of RDKit and xTB.
For full reproducibility, Windows users are strongly encouraged to use WSL2.

1) Install WSL2 with Ubuntu:

```python
wsl --install -d Ubuntu
```

2) Inside the WSL environment, install Conda (Miniconda or Mambaforge).

3) Clone the repository and create the environment:

```python
conda env create -f environment.linux.yml --name gleb_env
conda activate gleb_env
```

4) Install Python dependencies:

```python
python -m pip install -r requirements.linux.txt
python -m pip install git+https://github.com/IvanBespalov64/autogpr.git
```

This setup reproduces the same environment as the Linux server.

Windows (experimental, without WSL2)
----------------------
A native Windows installation may work, but is not officially supported and is provided without guarantees.
This configuration has been tested by the authors and is considered experimental.

Limitations:

RDKit API behavior may differ across versions.

xTB runtime behavior on Windows may be less stable than on Linux.

xTB-based descriptors may fail for some structures.

Experimental setup:

1) Create the experimental Conda environment:

```python
conda env create -f environment.windows.yml --name gleb_env
conda activate gleb_env
```

2) Install Python dependencies:

```python
python -m pip install -r requirements.windows.txt
python -m pip install git+https://github.com/IvanBespalov64/autogpr.git
```

If xTB-based descriptor calculation fails, users are advised to disable xTB features or switch to the WSL2-based setup.

Notes on reproducibility:

The Linux environment is the reference implementation.

All reported results were obtained using the Linux setup.

Windows support without WSL2 is provided on a best-effort basis.


Quick start:
-----------

```python
# 1) Featurization

# To featurize the monomer, you need SMILES with two *.
smiles = '*C(C(=O)N)(C*)C'

# Import the featurize function from the monomer_featurizer_predictor.py.
from monomer_featurizer_predictor import featurize
print(featurize(smiles)) # Returns a dictionary of feature : value.

# If you do not want to use xTB quantum-chemical package:
print(featurize(smiles, with_xtb = False)) # Returns the same dictionary with None for
# quantum-chemical descriptors. AND IT WORKS FASTER.

# You can generate a dictionary with full names of the descriptors:
from monomer_featurizer_predictor import featurize_to_dict
print(featurize_to_dict(smiles)) #or
print(featurize_to_dict(smiles, with_xtb = False))

# 2) Prediction

# To predict 4 properties of the corresponding polymer:
# - Install autogpr library.
# - Import the GPR models in the same directory as the monomer_featurizer_predictor.py file.
# - Use the predict_targets function:
from monomer_featurizer_predictor import predict_targets

# Generate a dataframe with mean and std of the predicted properties:
print(predict_targets(smiles))

# If you didn't install xTB for quantum-chemical calculations:
print(predict_targets(smiles, with_xtb = False)) # It will use models trained without xTB descriptors. AND IT WORKS FASTER.

# If you want to predict only one particular target:
print(predict_targets(smiles, target = 'Density')) # or target = 'Glass transition temperature'
# or 'Thermal decomposition temperature' or 'Melting temperature'
# returns a dictionary
```

