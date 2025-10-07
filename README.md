# Monomer Featurizer and Predictor of Polymer Properties

This script provides physically grounded and interpretable featurization
of monomers for Quantitative Structure–Property Relationship modeling.
It is designed for applications in polymer informatics and materials science.

Includes GPR models for prediction of polymer properties: density, glass transition temperature, thermal decomposition temperature, melting temperature.

Citation request:
------------------
If you use this code or models in your research, please cite the following paper:

Gleb Averochkin, Ivan Zlobin, Ivan Bespalov, Eugeniy Alexandrov, Machine Learning with Physically Grounded, Interpretable Descriptors for Polymer Property Prediction and Monomer Design, Chemical Engineering Journal

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
Import monomer_featurizer_predictor.py to generate descriptors from monomer SMILES. SMILES must have two polymerization sites, marked by * symbols.

The descriptors can be used as input features for machine learning models to predict polymer properties.

The monomer_featurizer_predictor.py file also contains functions for predicting the properties of corresponding polymers. To do it, you need to import the GPR models in the same directory.

Quick start:
-----------

```python
# 1) Featurization

# To featurize the monomer, you need SMILES with two *.
smiles = '*C(C(=O)N)(C*)C'

# Import the featurize function from the monomer_featurizer_predictor.py.
from monomer_featurizer_predictor import featurize

# For better performance you need to install the xTB quantum-chemical package. It is used to calculate descriptors.
# xtb version 6.7.1pre (5071a88) compiled by 'Marcel@Raven' on 2024-07-23

print(featurize(smiles)) # Returns a dictionary of feature : value.

# If you do not want to use xTB quantum-chemical package:
print(featurize(smiles, with_xtb = False)) # Returns the same dictionary with None for quantum-chemical descriptors.

# You can generate a dictionary with full names of the descriptors:
from monomer_featurizer_predictor import featurize_to_eng_dict
print(featurize_to_eng_dict(smiles)) #or
print(featurize_to_eng_dict(smiles, with_xtb = False))

# You can generate a dictionary with full names of the descriptors in Russian:
from monomer_featurizer_predictor import featurize_to_rus_dict
print(featurize_to_rus_dict(smiles)) #or
print(featurize_to_rus_dict(smiles, with_xtb = False))

# 2) Prediction

# To predict 4 properties of the corresponding polymer:
# - Import the GPR models in the same directory as the monomer_featurizer_predictor.py file.
# - Use the predict_targets function:
from monomer_featurizer_predictor import predict_targets

# Generate a dataframe with mean and std of the predicted properties:
print(predict_targets(smiles))

# If you didn't install xTB for quantum-chemical calculations:
print(predict_targets(smiles, with_xtb = False)) # It will use models trained without xTB descriptors

# If you want to predict only one particular target:
print(predict_targets(smiles, target = 'Density')) # or target = 'Glass transition temperature' or 'Thermal decomposition temperature' or 'Melting temperature'
# returns a dictionary
```

