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
Import the featurizer into your Python project to generate descriptors from monomer SMILES. SMILES must have two polymerization sites, marked by * symbols.
The descriptors can be used as input features for machine learning models to predict polymer properties.
The qspr_featurizer_predictor.py file also contains functions for predicting the properties of corresponding polymers. To do it, you need to import the GPR models in the same directory.

Quick start:
-----------


