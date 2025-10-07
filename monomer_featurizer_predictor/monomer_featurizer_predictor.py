"""
Monomer Featurizer and Predictor of Polymer Properties
=======================================

This script provides physically grounded and interpretable featurization
of monomers for Quantitative Structure–Property Relationship (QSPR) modeling.
It is designed for applications in polymer informatics and materials science.

Includes GPR models for prediction of polymer properties: density, glass transition temperature, thermal decomposition temperature, melting temperature.

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
Import the featurizer into your Python project to generate descriptors from SMILES. 
The descriptors can be used as input features for machine learning models to predict polymer properties.

License:
--------
Gleb Averochkin, Ivan Zlobin, Ivan Bespalov, Eugeniy Alexandrov, Machine Learning with Physically Grounded, Interpretable Descriptors for Polymer Property Prediction and Monomer Design, Chemical Engineering Journal

Author:
-------
Gleb Averochkin and contributors
"""


from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import GraphDescriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Crippen
import numpy as np
import math, tempfile, os, subprocess, re, requests, logging
from math import pi
import pandas as pd
import os
from pathlib import Path
import joblib
from autogpr.data import TabularData
from autogpr.model_selection import AutoGPRModel, GPRSelectionConfig
from typing import Optional, Union

atom_volumes_list = [
    'main_VdW_atoms_volumes',
    'main_h_donor_atoms_volumes',
    'main_dipole_atoms_volumes',
    'main_cation_atoms_volumes',
    'main_anion_atoms_volumes',
    'main_arhet_atoms_volumes',
    'main_arc_atoms_volumes',
    'side_VdW_atoms_volumes',
    'side_h_donor_atoms_volumes',
    'side_dipole_atoms_volumes',
    'side_cation_atoms_volumes',
    'side_anion_atoms_volumes',
    'side_arhet_atoms_volumes',
    'side_arc_atoms_volumes'
]





# Dictionary of van der Waals radii
vdw_radii = {
    "H": 1.1,
    "He": 1.4,
    "Li": 1.82,
    "Be": 1.53,
    "B": 1.92,
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "Ne": 1.54,
    "Na": 2.27,
    "Mg": 1.73,
    "Al": 1.84,
    "Si": 2.1,
    "P": 1.8,
    "S": 1.8,
    "Cl": 1.75,
    "Ar": 1.88,
    "K": 2.75,
    "Ca": 2.31,
    "Sc": 2.15,
    "Ti": 2.11,
    "V": 2.07,
    "Cr": 2.06,
    "Mn": 2.05,
    "Fe": 2.04,
    "Co": 2.0,
    "Ni": 1.97,
    "Cu": 1.96,
    "Zn": 2.01,
    "Ga": 1.87,
    "Ge": 2.11,
    "As": 1.85,
    "Se": 1.9,
    "Br": 1.85,
    "Kr": 2.02,
    "Rb": 3.03,
    "Sr": 2.49,
    "Y": 2.32,
    "Zr": 2.23,
    "Nb": 2.18,
    "Mo": 2.17,
    "Tc": 2.16,
    "Ru": 2.13,
    "Rh": 2.1,
    "Pd": 2.1,
    "Ag": 2.11,
    "Cd": 2.18,
    "In": 1.93,
    "Sn": 2.17,
    "Sb": 2.06,
    "Te": 2.06,
    "I": 1.98,
    "Xe": 2.16,
    "Cs": 3.43,
    "Ba": 2.68,
    "La": 2.43,
    "Ce": 2.42,
    "Pr": 2.4,
    "Nd": 2.39,
    "Pm": 2.38,
    "Sm": 2.36,
    "Eu": 2.35,
    "Gd": 2.34,
    "Tb": 2.33,
    "Dy": 2.31,
    "Ho": 2.3,
    "Er": 2.29,
    "Tm": 2.27,
    "Yb": 2.26,
    "Lu": 2.24,
    "Hf": 2.23,
    "Ta": 2.22,
    "W": 2.18,
    "Re": 2.16,
    "Os": 2.16,
    "Ir": 2.13,
    "Pt": 2.13,
    "Au": 2.14,
    "Hg": 2.23,
    "Tl": 1.96,
    "Pb": 2.02,
    "Bi": 2.07,
    "Po": 1.97,
    "At": 2.02,
    "Rn": 2.2,
    "Fr": 3.48,
    "Ra": 2.83,
    "Ac": 2.47,
    "Th": 2.45,
    "Pa": 2.43,
    "U": 2.41,
    "Np": 2.39,
    "Pu": 2.43,
    "Am": 2.44
}

ALLSMARTS = {'[CX4,c][OX2H]': 'hydroxyl',
             '[C,c][N](=O)[O]': 'nitro',
             '[O][N](=O)[O]': 'nitro_ester',
             '[NX3;H2;!n;!$(NC=O)]([CX4,c])': 'primary_amine',
             '[NX3;H1;!n;!$(NC=O)]([CX4,c])([CX4,c])': 'secondary_amine',
             '[NX3;H0;!n;!$(NC=O)]([CX4,c])([CX4,c])([CX4,c])': 'tertiary_amine',
             '[NX3H2][CX3](=[OX1])[#6]': 'primary_amide',
             '[NX3;H1;!$([NX3]([*](=O))([*](=O)))][CX3](=[OX1])[CX4,c]': 'secondary_amide',
             '[NX3;H0;!$([NX3](C(=O))(C(=O)))][CX3](=[OX1])[C,c]': 'tertiary_amide',
             '[NX1]#[CX2]': 'nitrile',
             '[NX3][CX3](=[OX1])[OX2]': 'carbamate',
             '[N][CX3](=[OX1])[N]': 'urea',
             '[CX3](=[OX1])[NX3][CX3](=[OX1])': 'imide',
             'c1ccc2c(c1)C(=O)NC2=O': 'phthalimide',
             '[n]1cccc1': 'pyrrole',
             '[n]1ccccc1': 'pyridine',
             '[NX3;H1,H2][c]1ccccc1': 'aniline',
             '[CX3](=O)[OX2H1]': 'carboxilic_acid',
             '[OD2]([CX4,c])[CX4,c]': 'ether',
             '[CX3](=O)[OX2][CX4,c]': 'ester',
             '[CX3H1](=O)[#6]': 'aldehyde',
             '[#5,#13,#31,#49,#81,#3,#11,#19,#37,#55,#87,#4,#12,#20,#38,#56,#88,#21,#39,#57,#89,#22,#40,#72,#104,#23,#41,#73,#105,#24,#42,#74,#106,#25,#43,#75,#107,#26,#44,#76,#108,#27,#45,#77,#109,#28,#46,#78,#110,#29,#47,#79,#111,#30,#48,#80,#112]': 'metal',
             '[CX3](=[OX1])[OX2][CX3](=[OX1])': 'anhydride',
             '[o]1cccc1': 'furan',
             '[O]1CCCCC1': 'tetrahydropyran',
             '[OH][c]1ccccc1': 'phenol',
             '[Na][O]': 'sodium_salt',
             '[#16X2H0]': 'sulfide',
             '[$([#16X4](=[OX1])(=[OX1])([#6])[#6]),$([#16X4+2]([OX1-])([OX1-])([#6])[#6])]': 'sulfone',
             '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]': 'sulfonate',
             '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]': 'sulfonamide',
             '[s]1cccc1': 'thiophene',
             '[P](=O)(O)(O)': 'po3_derivative',
             '[$(P(=[OX1])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)]),$([P+]([OX1-])([$([OX2H]),$([OX1-]),$([OX2]P)])([$([OX2H]),$([OX1-]),$([OX2]P)])[$([OX2H]),$([OX1-]),$([OX2]P)])]': 'phosphoric_acid',
             '[$(P(=[OX1])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)]),$([P+]([OX1-])([OX2][#6])([$([OX2H]),$([OX1-]),$([OX2][#6])])[$([OX2H]),$([OX1-]),$([OX2][#6]),$([OX2]P)])]': 'phosphoric_ester',
             '[P]([c]1ccccc1)([c]1ccccc1)([c]1ccccc1)': 'triphenylphosphine',
             '[Si]': 'silicon',
             '[Si]([C])([C])([C])[C]': 'sic4',
             '[Si]([C])([C])([C])[O]': 'sic3o',
             '[Si]([C])([C])([O])([O])': 'sic2o2',
             '[Si]([C])([O])([O])([O])': 'sico3',
             'c1ccccc1': 'benzene',
             '[CX4]': 'alyphatic_carbon',
             '[$([CX3]=[CX3])]': 'vinylic_carbon',
             '[$([CX2]#C)]': 'acetylenic_carbon',
            }



volumes = ['main_VdW_atoms_volumes', 'main_h_donor_atoms_volumes', 'main_dipole_atoms_volumes', 'main_cation_atoms_volumes', 'main_anion_atoms_volumes',
    'main_arhet_atoms_volumes', 'main_arc_atoms_volumes',
   'side_VdW_atoms_volumes', 'side_h_donor_atoms_volumes', 'side_dipole_atoms_volumes', 'side_cation_atoms_volumes', 'side_anion_atoms_volumes',
    'side_arhet_atoms_volumes', 'side_arc_atoms_volumes',
    'total_volume']

volume_types =     ['VdW', 'h_donor', 'dipole', 'cation', 'anion', 'arhet', 'arc']





all_chosen_features = [
     'Br_main_dipole_volume', 'Br_side_dipole_volume', 'C_main_arc_volume', 'C_main_VdW_volume', 
     'C_side_arc_volume', 'C_side_VdW_volume', 'Cl_main_dipole_volume', 'Cl_side_dipole_volume', 
     'F_main_dipole_volume', 'F_side_dipole_volume', 'FractionCSP3', 'H_main_h_donor_volume', 
     'H_main_VdW_volume', 'H_side_h_donor_volume', 'H_side_VdW_volume', 'HeavyAtomCount', 
     'N_main_dipole_volume', 'N_main_arhet_volume', 'N_main_h_donor_volume', 'N_side_dipole_volume', 
     'N_side_arhet_volume', 'N_side_h_donor_volume', 'Na_side_cation_volume', 'NumAliphaticCarbocycles', 
     'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 
     'NumAromaticRings', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 
     'NumSaturatedRings', 'O_main_dipole_volume', 'O_main_anion_volume', 'O_main_arhet_volume', 
     'O_main_h_donor_volume', 'O_side_dipole_volume', 'O_side_anion_volume', 'O_side_arhet_volume', 
     'O_side_h_donor_volume', 'P_main_dipole_volume', 'P_side_dipole_volume', 'RingCount', 
     'S_main_dipole_volume', 'S_main_arhet_volume', 'S_side_dipole_volume', 'S_side_arhet_volume', 
     'Si_main_dipole_volume', 'Si_side_dipole_volume', 'anion_feature_main', 'anion_feature_side', 
     'acetylenic_carbon_main', 'acetylenic_carbon_side', 'alyphatic_carbon_main', 'alyphatic_carbon_side', 
     'anhydride_main', 'anhydride_side', 'aniline_main', 'aniline_side', 'arc_feature_main', 
     'arc_feature_side', 'arhet_feature_main', 'arhet_feature_side', 'balaban_index', 'benzene_main', 
     'benzene_side', 'cation_feature_main', 'cation_feature_side', 'carbamate_main', 'carbamate_side', 
     'carboxilic_acid_main','carboxilic_acid_side', 'ester_main', 'ester_side', 'ether_main', 'ether_side', 
     'rotatable_bond_fraction', 'furan_main', 'furan_side', 'h_donor_feature_main', 'h_donor_feature_side', 'hallkier_alpha', 
     'hydroxyl_main', 'hydroxyl_side', 'imide_main', 'imide_side', 'log_p', 'main_aromatic_bond_fraction', 
     'main_chain_volume', 'main_double_bond_fraction', 'main_single_bond_fraction', 'main_triple_bond_fraction', 
     'metal_main', 'metal_side', 'mol_mass', 'nitrile_main', 'nitrile_side', 'nitro_ester_main', 
     'nitro_ester_side', 'nitro_main', 'nitro_side', 'dipole_feature_main', 'dipole_feature_side', 
     'phenol_main', 'phenol_side', 'phosphoric_ester_main', 'phosphoric_ester_side', 'phthalimide_main', 
     'phthalimide_side', 'po3_derivative_main', 'po3_derivative_side', 'phosphoric_acid_main', 'phosphoric_acid_side', 'primary_amide_main', 
     'primary_amide_side', 'primary_amine_main', 'primary_amine_side', 'pyridine_main', 
     'pyridine_side', 'pyrrole_main', 'pyrrole_side', 'secondary_amide_main', 
     'secondary_amide_side', 'secondary_amine_main', 'secondary_amine_side', 'sic2o2_main', 'sic2o2_side',
     'sic3o_main', 'sic3o_side', 'sic4_main', 'sic4_side', 'side_aromatic_bond_fraction', 'side_chain_volume', 
     'side_double_bond_fraction', 'side_single_bond_fraction', 'side_triple_bond_fraction', 'silicon_main', 
     'silicon_side', 'sodium_salt_main', 'sodium_salt_side', 'sulfide_main', 'sulfide_side', 'sulfonamide_main', 
     'sulfonamide_side', 'sulfonate_main', 'sulfonate_side', 'sulfone_main', 'sulfone_side', 
     'tertiary_amide_main', 'tertiary_amide_side', 'tertiary_amine_main', 'tertiary_amine_side', 
     'tetrahydropyran_main', 'tetrahydropyran_side', 'thiophene_main', 'thiophene_side', 'tpsa', 'triphenylphosphine_main', 'triphenylphosphine_side',
     'urea_main', 'urea_side', 'vinylic_carbon_main', 'vinylic_carbon_side',
     'H_main', 'H_side',
     'Cl_main', 'Cl_side',
     'Br_main', 'Br_side',
     'F_main', 'F_side',
     'distance_between_polymerization_sites',
     "total_energy (hartree)",
     "homo_lumo_gap (eV)",
     "homo_energy (eV)",
     "lumo_energy (eV)",
     "dipole_moment (D)",
     "mulliken_charge_std (e)",
     "mulliken_charge_min (e)",
     "mulliken_charge_max (e)"
]

all_chosen_features_rus = [
    'ВдВ объем главной цепи, занимаемый атомами брома (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами брома (Å³)',
    'ВдВ объем главной цепи, занимаемый ароматическими атомами углерода (Å³)',
    'ВдВ объем главной цепи, занимаемый атомами углерода с преимущественным ВдВ межмолекулярным взаимодействием (Å³)', 
    'ВдВ объем боковой цепи, занимаемый ароматическими атомами углерода (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами углерода с преимущественным ВдВ межмолекулярным взаимодействием (Å³)',
    'ВдВ объем главной цепи, занимаемый атомами хлора (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами хлора (Å³)', 
    'ВдВ объем главной цепи, занимаемый атомами фтора (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами фтора (Å³)',
    'Доля sp3-гибридизованных атомов углерода',
    'ВдВ объем главной цепи, занимаемый атомами водорода - инициаторами водородных связей (Å³)', 
    'ВдВ объем главной цепи, занимаемый атомами водорода с преимущественным ВдВ межмолекулярным взаимодействием (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами водорода - инициаторами водородных связей (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами водорода с преимущественным ВдВ межмолекулярным взаимодействием (Å³)',
    'Количество тяжелых атомов', 
    'ВдВ объем главной цепи, занимаемый атомами азота с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)',
    'ВдВ объем главной цепи, занимаемый ароматическими атомами азота (Å³)',
    'ВдВ объем главной цепи, занимаемый атомами азота, связанными с водородом (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами азота с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)', 
    'ВдВ объем боковой цепи, занимаемый ароматическими атомами азота (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами азота, связанными с водородом (Å³)',
    'ВдВ объем боковой цепи, занимаемый катионами натрия (Å³)',
    'Количество алифатических карбоциклов', 
    'Количество алифатических гетероциклов',
    'Количество алифатических циклов', 
    'Количество ароматических карбоциклов',
    'Количество ароматических гетероциклов', 
    'Количество ароматических колец',
    'Количество гетероатомов',
    'Число вращаемых связей', 
    'Количество насыщенных карбоциклов',
    'Количество насыщенных гетероциклов', 
    'Количество насыщенных колец',
    'ВдВ объем главной цепи, занимаемый атомами кислорода с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)',
    'ВдВ объем главной цепи, занимаемый отрицательно заряженными атомами кислорода (Å³)',
    'ВдВ объем главной цепи, занимаемый ароматическими атомами кислорода (Å³)', 
    'ВдВ объем главной цепи, занимаемый атомами кислорода, связанными с водородом (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами кислорода с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)',
    'ВдВ объем боковой цепи, занимаемый отрицательно заряженными атомами кислорода (Å³)',
    'ВдВ объем боковой цепи, занимаемый ароматическими атомами кислорода (Å³)', 
    'ВдВ объем боковой цепи, занимаемый атомами кислорода, связанными с водородом (Å³)',
    'ВдВ объем главной цепи, занимаемый атомами фосфора (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами фосфора (Å³)',
    'Количество колец', 
    'ВдВ объем главной цепи, занимаемый атомами серы с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)',
    'ВдВ объем главной цепи, занимаемый ароматическими атомами серы (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами серы с преимущественным диполь-дипольным межмолекулярным взаимодействием (Å³)',
    'ВдВ объем боковой цепи, занимаемый ароматическими атомами серы (Å³)', 
    'ВдВ объем главной цепи, занимаемый атомами кремния (Å³)',
    'ВдВ объем боковой цепи, занимаемый атомами кремния (Å³)',
    'Суммарная доля ВдВ объема главной цепи, занимаемого отрицательно заряженными атомами',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого отрицательно заряженными атомами', 
    'Количество sp-гибридизованных атомов углерода в главной цепи',
    'Количество sp-гибридизованных атомов углерода в боковой цепи',
    'Количество sp3-гибридизованных атомов углерода в главной цепи',
    'Количество sp3-гибридизованных атомов углерода в боковой цепи', 
    'Количество фрагментов ангидридов в главной цепи',
    'Количество фрагментов ангидридов в боковой цепи', 
    'Количество фрагментов анилинов в главной цепи',
    'Количество фрагментов анилинов в боковой цепи', 
    'Суммарная доля ВдВ объема главной цепи, занимаемого ароматическими атомами углерода',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого ароматическими атомами углерода',
    'Суммарная доля ВдВ объема главной цепи, занимаемого ароматическими гетероатомами',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого ароматическими гетероатомами',
    'Индекс Балабана', 
    'Количество фрагментов бензола в главной цепи',
    'Количество фрагментов бензола в боковой цепи', 
    'Суммарная доля ВдВ объема главной цепи, занимаемого положительно заряженными атомами',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого положительно заряженными атомами',
    'Количество фрагментов карбаматов в главной цепи',
    'Количество фрагментов карбаматов в боковой цепи',
    'Количество фрагментов карбоновых кислот в главной цепи',
    'Количество фрагментов карбоновых кислот в боковой цепи', 
    'Количество фрагментов сложных эфиров в главной цепи',
    'Количество фрагментов сложных эфиров в боковой цепи', 
    'Количество фрагментов простых эфиров в главной цепи',
    'Количество фрагментов простых эфиров в боковой цепи', 
    'Доля вращаемых связей', 
    'Количество фрагментов фуранов в главной цепи', 
    'Количество фрагментов фуранов в боковой цепи',
    'Суммарная доля ВдВ объема главной цепи, занимаемого функциональными группами, инициирующими водородные связи',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого функциональными группами, инициирующими водородные связи', 
    'Параметр HallkierAlpha', 
    'Количество гидроксильных групп в главной цепи',
    'Количество гидроксильных групп в боковой цепи',
    'Количество фрагментов имидов в главной цепи',
    'Количество фрагментов имидов в боковой цепи', 
    'Логарифм коэффициента распределения (logP)', 
    'Доля ароматических связей в главной цепи', 
    'ВдВ объем главной цепи (Å³)',
    'Доля двойных связей в главной цепи',
    'Доля одинарных связей в главной цепи',
    'Доля тройных связей в главной цепи', 
    'Количество атомов металлов в главной цепи', 
    'Количество атомов металлов в боковой цепи', 
    'Молекулярная масса (г/моль)', 
    'Количество фрагментов нитрилов в главной цепи',
    'Количество фрагментов нитрилов в боковой цепи', 
    'Количество фрагментов нитроэфиров в главной цепи',
    'Количество фрагментов нитроэфиров в боковой цепи', 
    'Количество нитро-групп в главной цепи',
    'Количество нитро-групп в боковой цепи', 
    'Суммарная доля ВдВ объема главной цепи, занимаемого атомами с примущественно диполь-дипольным межмолекулярным взаимодействием',
    'Суммарная доля ВдВ объема боковой цепи, занимаемого атомами с примущественно диполь-дипольным межмолекулярным взаимодействием', 
    'Количество фрагментов фенолов в главной цепи',
    'Количество фрагментов фенолов в боковой цепи',
    'Количество фрагментов фосфорных эфиров в главной цепи',
    'Количество фрагментов фосфорных эфиров в боковой цепи',
    'Количество фрагментов фталимидов в главной цепи',
    'Количество фрагментов фталимидов в боковой цепи', 
    'Количество фрагментов PO3-производных в главной цепи',
    'Количество фрагментов PO3-производных в боковой цепи',
    'Количество фрагментов фосфорной кислоты в главной цепи',
    'Количество фрагментов фосфорной кислоты в боковой цепи',
    'Количество фрагментов первичных амидов в главной цепи',
    'Количество фрагментов первичных амидов в боковой цепи', 
    'Количество фрагментов первичных аминов в главной цепи',
    'Количество фрагментов первичных аминов в боковой цепи', 
    'Количество фрагментов пиридина в главной цепи',
    'Количество фрагментов пиридина в боковой цепи', 
    'Количество фрагментов пиррола в главной цепи',
    'Количество фрагментов пиррола в боковой цепи', 
    'Количество фрагментов вторичных амидов в главной цепи',
    'Количество фрагментов вторичных амидов в боковой цепи', 
    'Количество фрагментов вторичных аминов в главной цепи',
    'Количество фрагментов вторичных аминов в боковой цепи', 
    'Количество фрагментов производных SiC2O2 в главной цепи',
    'Количество фрагментов производных SiC2O2 в боковой цепи',
    'Количество фрагментов производных SiC3O в главной цепи', 
    'Количество фрагментов производных SiC3O в боковой цепи',
    'Количество фрагментов производных SiC4 в главной цепи', 
    'Количество фрагментов производных SiC4 в боковой цепи', 
    'Доля ароматических связей в боковой цепи', 
    'ВдВ объем боковой цепи (Å³)', 
    'Доля двойных связей в боковой цепи',
    'Доля одинарных связей в боковой цепи',
    'Доля тройных связей в боковой цепи', 
    'Количество атомов кремния в главной цепи',
    'Количество атомов кремния в боковой цепи',
    'Количество фрагментов солей натрия в главной цепи',
    'Количество фрагментов солей натрия в боковой цепи', 
    'Количество фрагментов сульфидов в главной цепи',
    'Количество фрагментов сульфидов в боковой цепи', 
    'Количество фрагментов сульфонамидов в главной цепи',
    'Количество фрагментов сульфонамидов в боковой цепи',
    'Количество фрагментов сульфонатов в главной цепи',
    'Количество фрагментов сульфонатов в боковой цепи', 
    'Количество фрагментов сульфонов в главной цепи',
    'Количество фрагментов сульфонов в боковой цепи', 
    'Количество фрагментов третичных амидов в главной цепи',
    'Количество фрагментов третичных амидов в боковой цепи', 
    'Количество фрагментов третичных аминов в главной цепи',
    'Количество фрагментов третичных аминов в боковой цепи', 
    'Количество фрагментов тетрагидропиранов в главной цепи',
    'Количество фрагментов тетрагидропиранов в боковой цепи',
    'Количество фрагментов тиофенов в главной цепи',
    'Количество фрагментов тиофенов в боковой цепи', 
    'Топологическая полярная поверхность (Å²)', 
    'Количество фрагментов трифенилфосфина в главной цепи',
    'Количество фрагментов трифенилфосфина в боковой цепи',
    'Количество фрагментов мочевин в главной цепи',
    'Количество фрагментов мочевин в боковой цепи', 
    'Количество sp2-гибридизованных атомов углерода в главной цепи',
    'Количество sp2-гибридизованных атомов углерода в боковой цепи',
    'Количество атомов водорода в главной цепи',
    'Количество атомов водорода в боковой цепи',
    'Количество атомов хлора в главной цепи',
    'Количество атомов хлора в боковой цепи',
    'Количество атомов брома в главной цепи',
    'Количество атомов брома в боковой цепи',
    'Количество атомов фтора в главной цепи',
    'Количество атомов фтора в боковой цепи',
    'Расстояние между сайтами полимеризации (Å)',
    "Полная энергия молекулы мономера (Хартри)",
    "Энергетический зазор между ВЗМО и НСМО (эВ)",
    "Энергия ВЗМО (эВ)",
    "Энергия НСМО (эВ)",
    "Модуль дипольного момента молекулы мономера (D)",
    "Стандартное отклонение зарядов Малликена на атомах мономера (е)",
    "Минимальный заряд Малликена среди атомов мономера (е)",
    "Максимальный заряд Малликена среди атомов мономера (е)"
]

all_chosen_features_eng = [
    "VdW volume of main chain, occupied by bromine atoms (Å³)",
    "VdW volume of side chain, occupied by bromine atoms (Å³)",
    "VdW volume of main chain, occupied by aromatic carbon atoms (Å³)",
    "VdW volume of main chain, occupied by carbon atoms with predominant VdW intermolecular interaction (Å³)",
    "VdW volume of side chain, occupied by aromatic carbon atoms (Å³)",
    "VdW volume of side chain, occupied by carbon atoms with predominant VdW intermolecular interaction (Å³)",
    "VdW volume of main chain, occupied by chlorine atoms (Å³)",
    "VdW volume of side chain, occupied by chlorine atoms (Å³)",
    "VdW volume of main chain, occupied by fluorine atoms (Å³)",
    "VdW volume of side chain, occupied by fluorine atoms (Å³)",
    "Fraction of sp3-hybridized carbon atoms",
    "VdW volume of main chain, occupied by hydrogen atoms initiating hydrogen bonds (Å³)",
    "VdW volume of main chain, occupied by hydrogen atoms with predominant VdW intermolecular interaction (Å³)",
    "VdW volume of side chain, occupied by hydrogen atoms initiating hydrogen bonds (Å³)",
    "VdW volume of side chain, occupied by hydrogen atoms with predominant VdW intermolecular interaction (Å³)",
    "Number of heavy atoms",
    "VdW volume of main chain, occupied by nitrogen atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of main chain, occupied by aromatic nitrogen atoms (Å³)",
    "VdW volume of main chain, occupied by nitrogen atoms bonded to hydrogen (Å³)",
    "VdW volume of side chain, occupied by nitrogen atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of side chain, occupied by aromatic nitrogen atoms (Å³)",
    "VdW volume of side chain, occupied by nitrogen atoms bonded to hydrogen (Å³)",
    "VdW volume of side chain, occupied by sodium cations (Å³)",
    "Number of aliphatic carbocycles",
    "Number of aliphatic heterocycles",
    "Number of aliphatic rings",
    "Number of aromatic carbocycles",
    "Number of aromatic heterocycles",
    "Number of aromatic rings",
    "Number of heteroatoms",
    "Number of rotatable bonds",
    "Number of saturated carbocycles",
    "Number of saturated heterocycles",
    "Number of saturated rings",
    "VdW volume of main chain, occupied by oxygen atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of main chain, occupied by negatively charged oxygen atoms (Å³)",
    "VdW volume of main chain, occupied by aromatic oxygen atoms (Å³)",
    "VdW volume of main chain, occupied by oxygen atoms bonded to hydrogen (Å³)",
    "VdW volume of side chain, occupied by oxygen atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of side chain, occupied by negatively charged oxygen atoms (Å³)",
    "VdW volume of side chain, occupied by aromatic oxygen atoms (Å³)",
    "VdW volume of side chain, occupied by oxygen atoms bonded to hydrogen (Å³)",
    "VdW volume of main chain, occupied by phosphorus atoms (Å³)",
    "VdW volume of side chain, occupied by phosphorus atoms (Å³)",
    "Number of rings",
    "VdW volume of main chain, occupied by sulfur atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of main chain, occupied by aromatic sulfur atoms (Å³)",
    "VdW volume of side chain, occupied by sulfur atoms with predominant dipole-dipole intermolecular interaction (Å³)",
    "VdW volume of side chain, occupied by aromatic sulfur atoms (Å³)",
    "VdW volume of main chain, occupied by silicon atoms (Å³)",
    "VdW volume of side chain, occupied by silicon atoms (Å³)",
    "Fraction of VdW volume of main chain occupied by negatively charged atoms",
    "Fraction of VdW volume of side chain occupied by negatively charged atoms",
    "Number of sp-hybridized carbon atoms in main chain",
    "Number of sp-hybridized carbon atoms in side chain",
    "Number of sp3-hybridized carbon atoms in main chain",
    "Number of sp3-hybridized carbon atoms in side chain",
    "Number of anhydride fragments in main chain",
    "Number of anhydride fragments in side chain",
    "Number of aniline fragments in main chain",
    "Number of aniline fragments in side chain",
    "Fraction of VdW volume of main chain occupied by aromatic carbon atoms",
    "Fraction of VdW volume of side chain occupied by aromatic carbon atoms",
    "Fraction of VdW volume of main chain occupied by aromatic heteroatoms",
    "Fraction of VdW volume of side chain occupied by aromatic heteroatoms",
    "Balaban index",
    "Number of benzene fragments in main chain",
    "Number of benzene fragments in side chain",
    "Fraction of VdW volume of main chain occupied by positively charged atoms",
    "Fraction of VdW volume of side chain occupied by positively charged atoms",
    "Number of carbamate fragments in main chain",
    "Number of carbamate fragments in side chain",
    "Number of carboxylic acid fragments in main chain",
    "Number of carboxylic acid fragments in side chain",
    "Number of ester fragments in main chain",
    "Number of ester fragments in side chain",
    "Number of ether fragments in main chain",
    "Number of ether fragments in side chain",
    "Fraction of rotatable bonds",
    "Number of furan fragments in main chain",
    "Number of furan fragments in side chain",
    "Fraction of VdW volume of main chain occupied by H-bond donor groups",
    "Fraction of VdW volume of side chain occupied by H-bond donor groups",
    "HallkierAlpha parameter",
    "Number of hydroxyl groups in main chain",
    "Number of hydroxyl groups in side chain",
    "Number of imide fragments in main chain",
    "Number of imide fragments in side chain",
    "LogP",
    "Fraction of aromatic bonds in main chain",
    "VdW volume of main chain (Å³)",
    "Fraction of double bonds in main chain",
    "Fraction of single bonds in main chain",
    "Fraction of triple bonds in main chain",
    "Number of metal atoms in main chain",
    "Number of metal atoms in side chain",
    "Molecular weight (g/mol)",
    "Number of nitrile fragments in main chain",
    "Number of nitrile fragments in side chain",
    "Number of nitro ester fragments in main chain",
    "Number of nitro ester fragments in side chain",
    "Number of nitro groups in main chain",
    "Number of nitro groups in side chain",
    "Fraction of VdW volume of main chain occupied by atoms with predominant dipole-dipole intermolecular interaction",
    "Fraction of VdW volume of side chain occupied by atoms with predominant dipole-dipole intermolecular interaction",
    "Number of phenol fragments in main chain",
    "Number of phenol fragments in side chain",
    "Number of phosphate ester fragments in main chain",
    "Number of phosphate ester fragments in side chain",
    "Number of phthalimide fragments in main chain",
    "Number of phthalimide fragments in side chain",
    "Number of PO3-derivative fragments in main chain",
    "Number of PO3-derivative fragments in side chain",
    "Number of phosphoric acid fragments in main chain",
    "Number of phosphoric acid fragments in side chain",
    "Number of primary amide fragments in main chain",
    "Number of primary amide fragments in side chain",
    "Number of primary amine fragments in main chain",
    "Number of primary amine fragments in side chain",
    "Number of pyridine fragments in main chain",
    "Number of pyridine fragments in side chain",
    "Number of pyrrole fragments in main chain",
    "Number of pyrrole fragments in side chain",
    "Number of secondary amide fragments in main chain",
    "Number of secondary amide fragments in side chain",
    "Number of secondary amine fragments in main chain",
    "Number of secondary amine fragments in side chain",
    "Number of SiC2O2-derivative fragments in main chain",
    "Number of SiC2O2-derivative fragments in side chain",
    "Number of SiC3O-derivative fragments in main chain",
    "Number of SiC3O-derivative fragments in side chain",
    "Number of SiC4-derivative fragments in main chain",
    "Number of SiC4-derivative fragments in side chain",
    "Fraction of aromatic bonds in side chain",
    "VdW volume of side chain (Å³)",
    "Fraction of double bonds in side chain",
    "Fraction of single bonds in side chain",
    "Fraction of triple bonds in side chain",
    "Number of silicon atoms in main chain",
    "Number of silicon atoms in side chain",
    "Number of sodium salt fragments in main chain",
    "Number of sodium salt fragments in side chain",
    "Number of sulfide fragments in main chain",
    "Number of sulfide fragments in side chain",
    "Number of sulfonamide fragments in main chain",
    "Number of sulfonamide fragments in side chain",
    "Number of sulfonate fragments in main chain",
    "Number of sulfonate fragments in side chain",
    "Number of sulfone fragments in main chain",
    "Number of sulfone fragments in side chain",
    "Number of tertiary amide fragments in main chain",
    "Number of tertiary amide fragments in side chain",
    "Number of tertiary amine fragments in main chain",
    "Number of tertiary amine fragments in side chain",
    "Number of tetrahydropyran fragments in main chain",
    "Number of tetrahydropyran fragments in side chain",
    "Number of thiophene fragments in main chain",
    "Number of thiophene fragments in side chain",
    "Topological polar surface area (Å²)",
    "Number of triphenylphosphine fragments in main chain",
    "Number of triphenylphosphine fragments in side chain",
    "Number of urea fragments in main chain",
    "Number of urea fragments in side chain",
    "Number of sp2-hybridized carbon atoms in main chain",
    "Number of sp2-hybridized carbon atoms in side chain",
    "Number of hydrogen atoms in main chain",
    "Number of hydrogen atoms in side chain",
    "Number of chlorine atoms in main chain",
    "Number of chlorine atoms in side chain",
    "Number of bromine atoms in main chain",
    "Number of bromine atoms in side chain",
    "Number of fluorine atoms in main chain",
    "Number of fluorine atoms in side chain",
    "Distance between polymerization sites (Å)",
    "Total energy of monomer molecule (Hartree)",
    "HOMO-LUMO energy gap (eV)",
    "HOMO energy (eV)",
    "LUMO energy (eV)",
    "Dipole moment magnitude of monomer molecule (D)",
    "Standard deviation of Mulliken charges on monomer atoms (e)",
    "Minimum Mulliken charge among monomer atoms (e)",
    "Maximum Mulliken charge among monomer atoms (e)"
]

features_by_target = {'Density': ['dipole_feature_side',
  'FractionCSP3',
  'mulliken_charge_std (e)',
  'lumo_energy (eV)',
  'homo_energy (eV)',
  'NumHeteroatoms',
  'F_main_dipole_volume',
  'dipole_feature_main',
  'H_side_VdW_volume',
  'dipole_moment (D)',
  'rotatable_bond_fraction',
  'log_p',
  'mulliken_charge_min (e)',
  'balaban_index',
  'C_side_VdW_volume',
  'homo_lumo_gap (eV)',
  'hallkier_alpha',
  'mulliken_charge_max (e)',
  'H_side',
  'C_main_VdW_volume',
  'main_chain_volume',
  'tpsa',
  'side_chain_volume',
  'main_double_bond_fraction',
  'alyphatic_carbon_main',
  'side_single_bond_fraction',
  'H_main_VdW_volume',
  'alyphatic_carbon_side',
  'Br_side_dipole_volume',
  'mol_mass',
  'arc_feature_main',
  'NumRotatableBonds',
  'distance_between_polymerization_sites',
  'O_side_dipole_volume',
  'h_donor_feature_side',
  'arhet_feature_main',
  'O_main_dipole_volume',
  'arc_feature_side',
  'h_donor_feature_main',
  'C_side_arc_volume'],
 'Glass transition temperature': ['rotatable_bond_fraction',
  'RingCount',
  'FractionCSP3',
  'N_main_h_donor_volume',
  'hallkier_alpha',
  'dipole_feature_main',
  'arc_feature_main',
  'tpsa',
  'homo_lumo_gap (eV)',
  'mulliken_charge_max (e)',
  'C_main_arc_volume',
  'secondary_amide_main',
  'balaban_index',
  'lumo_energy (eV)',
  'Si_main_dipole_volume',
  'mulliken_charge_min (e)',
  'h_donor_feature_main',
  'homo_energy (eV)',
  'distance_between_polymerization_sites',
  'H_main_VdW_volume',
  'dipole_moment (D)',
  'log_p',
  'aniline_main',
  'side_chain_volume',
  'main_double_bond_fraction',
  'mulliken_charge_std (e)',
  'C_main_VdW_volume',
  'main_chain_volume',
  'mol_mass',
  'H_side_VdW_volume',
  'O_main_dipole_volume',
  'O_side_dipole_volume',
  'arc_feature_side',
  'dipole_feature_side',
  'C_side_arc_volume',
  'C_side_VdW_volume',
  'benzene_main',
  'H_side_h_donor_volume',
  'H_side',
  'ester_main',
  'alyphatic_carbon_main',
  'N_main_dipole_volume',
  'NumHeteroatoms',
  'side_single_bond_fraction',
  'H_main',
  'h_donor_feature_side',
  'NumRotatableBonds',
  'aniline_side',
  'side_aromatic_bond_fraction',
  'arhet_feature_main'],
 'Thermal decomposition temperature': ['arc_feature_main',
  'rotatable_bond_fraction',
  'mulliken_charge_min (e)',
  'homo_energy (eV)',
  'phthalimide_main',
  'mulliken_charge_max (e)',
  'main_double_bond_fraction',
  'lumo_energy (eV)',
  'homo_lumo_gap (eV)',
  'side_single_bond_fraction',
  'log_p',
  'dipole_feature_main',
  'mulliken_charge_std (e)',
  'dipole_moment (D)',
  'side_chain_volume',
  'balaban_index',
  'C_main_VdW_volume',
  'tpsa',
  'hallkier_alpha',
  'C_side_VdW_volume',
  'distance_between_polymerization_sites',
  'H_main_VdW_volume',
  'C_main_arc_volume',
  'main_chain_volume',
  'arc_feature_side',
  'Si_main_dipole_volume',
  'h_donor_feature_main',
  'O_main_dipole_volume',
  'mol_mass',
  'dipole_feature_side',
  'FractionCSP3',
  'H_side_VdW_volume',
  'C_side_arc_volume',
  'O_side_dipole_volume',
  'ester_main',
  'benzene_main',
  'N_main_dipole_volume',
  'side_double_bond_fraction',
  'furan_side',
  'NumSaturatedHeterocycles',
  'h_donor_feature_side',
  'S_main_dipole_volume',
  'N_side_dipole_volume',
  'NumHeteroatoms',
  'H_side',
  'P_main_dipole_volume',
  'O_side_h_donor_volume',
  'arhet_feature_main',
  'N_main_h_donor_volume',
  'acetylenic_carbon_main',
  'H_main',
  'N_main_arhet_volume',
  'tertiary_amine_side',
  'alyphatic_carbon_main',
  'RingCount',
  'NumRotatableBonds',
  'po3_derivative_main',
  'pyridine_main',
  'O_main_arhet_volume',
  'carbamate_main',
  'H_side_h_donor_volume',
  'NumAromaticHeterocycles',
  'NumAromaticRings',
  'thiophene_main',
  'arhet_feature_side',
  'furan_main',
  'side_aromatic_bond_fraction',
  'urea_main',
  'S_main_arhet_volume',
  'secondary_amide_main'],
 'Melting temperature': ['rotatable_bond_fraction',
  'FractionCSP3',
  'hallkier_alpha',
  'arc_feature_main',
  'tpsa',
  'C_main_arc_volume',
  'mulliken_charge_max (e)',
  'RingCount',
  'NumAromaticRings',
  'lumo_energy (eV)',
  'benzene_main',
  'alyphatic_carbon_main',
  'C_main_VdW_volume',
  'homo_lumo_gap (eV)',
  'balaban_index',
  'O_main_dipole_volume',
  'mulliken_charge_min (e)',
  'side_chain_volume',
  'H_side',
  'H_main',
  'N_main_h_donor_volume',
  'dipole_feature_main',
  'H_side_VdW_volume',
  'h_donor_feature_main',
  'C_side_arc_volume',
  'main_double_bond_fraction',
  'H_main_VdW_volume',
  'homo_energy (eV)',
  'NumRotatableBonds',
  'secondary_amide_main']}

features_by_target_no_xtb = {'Density': ['C_main_arc_volume',
 'C_main_VdW_volume',
 'C_side_arc_volume',
 'C_side_VdW_volume',
 'F_side_dipole_volume',
 'FractionCSP3',
 'H_main_h_donor_volume',
 'H_main_VdW_volume',
 'H_side_h_donor_volume',
 'H_side_VdW_volume',
 'N_main_dipole_volume',
 'N_main_arhet_volume',
 'N_side_dipole_volume',
 'N_side_arhet_volume',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'O_main_dipole_volume',
 'O_side_dipole_volume',
 'O_side_h_donor_volume',
 'RingCount',
 'S_main_dipole_volume',
 'S_main_arhet_volume',
 'Si_main_dipole_volume',
 'alyphatic_carbon_main',
 'alyphatic_carbon_side',
 'aniline_main',
 'arc_feature_main',
 'arc_feature_side',
 'arhet_feature_main',
 'arhet_feature_side',
 'balaban_index',
 'benzene_main',
 'benzene_side',
 'carbamate_main',
 'ester_main',
 'ester_side',
 'ether_main',
 'ether_side',
 'rotatable_bond_fraction',
 'h_donor_feature_main',
 'h_donor_feature_side',
 'hallkier_alpha',
 'imide_main',
 'log_p',
 'main_chain_volume',
 'main_double_bond_fraction',
 'mol_mass',
 'dipole_feature_main',
 'dipole_feature_side',
 'phthalimide_main',
 'pyrrole_main',
 'secondary_amide_main',
 'side_aromatic_bond_fraction',
 'side_chain_volume',
 'side_double_bond_fraction',
 'side_single_bond_fraction',
 'sulfide_main',
 'sulfone_main',
 'tpsa',
 'vinylic_carbon_main',
 'H_main',
 'H_side',
 'distance_between_polymerization_sites'],
 'Glass transition temperature': ['C_main_arc_volume',
 'C_main_VdW_volume',
 'C_side_arc_volume',
 'C_side_VdW_volume',
 'F_side_dipole_volume',
 'FractionCSP3',
 'H_main_h_donor_volume',
 'H_main_VdW_volume',
 'H_side_h_donor_volume',
 'H_side_VdW_volume',
 'N_main_dipole_volume',
 'N_main_arhet_volume',
 'N_side_dipole_volume',
 'N_side_arhet_volume',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'O_main_dipole_volume',
 'O_side_dipole_volume',
 'O_side_h_donor_volume',
 'RingCount',
 'S_main_dipole_volume',
 'S_main_arhet_volume',
 'Si_main_dipole_volume',
 'alyphatic_carbon_main',
 'alyphatic_carbon_side',
 'aniline_main',
 'arc_feature_main',
 'arc_feature_side',
 'arhet_feature_main',
 'arhet_feature_side',
 'balaban_index',
 'benzene_main',
 'benzene_side',
 'carbamate_main',
 'ester_main',
 'ester_side',
 'ether_main',
 'ether_side',
 'rotatable_bond_fraction',
 'h_donor_feature_main',
 'h_donor_feature_side',
 'hallkier_alpha',
 'imide_main',
 'log_p',
 'main_chain_volume',
 'main_double_bond_fraction',
 'mol_mass',
 'dipole_feature_main',
 'dipole_feature_side',
 'phthalimide_main',
 'pyrrole_main',
 'secondary_amide_main',
 'side_aromatic_bond_fraction',
 'side_chain_volume',
 'side_double_bond_fraction',
 'side_single_bond_fraction',
 'sulfide_main',
 'sulfone_main',
 'tpsa',
 'vinylic_carbon_main',
 'H_main',
 'H_side',
 'distance_between_polymerization_sites'],
 'Thermal decomposition temperature': ['C_main_arc_volume',
 'C_main_VdW_volume',
 'C_side_arc_volume',
 'C_side_VdW_volume',
 'F_side_dipole_volume',
 'FractionCSP3',
 'H_main_h_donor_volume',
 'H_main_VdW_volume',
 'H_side_h_donor_volume',
 'H_side_VdW_volume',
 'N_main_dipole_volume',
 'N_main_arhet_volume',
 'N_side_dipole_volume',
 'N_side_arhet_volume',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'O_main_dipole_volume',
 'O_side_dipole_volume',
 'O_side_h_donor_volume',
 'RingCount',
 'S_main_dipole_volume',
 'S_main_arhet_volume',
 'Si_main_dipole_volume',
 'alyphatic_carbon_main',
 'alyphatic_carbon_side',
 'aniline_main',
 'arc_feature_main',
 'arc_feature_side',
 'arhet_feature_main',
 'arhet_feature_side',
 'balaban_index',
 'benzene_main',
 'benzene_side',
 'carbamate_main',
 'ester_main',
 'ester_side',
 'ether_main',
 'ether_side',
 'rotatable_bond_fraction',
 'h_donor_feature_main',
 'h_donor_feature_side',
 'hallkier_alpha',
 'imide_main',
 'log_p',
 'main_chain_volume',
 'main_double_bond_fraction',
 'mol_mass',
 'dipole_feature_main',
 'dipole_feature_side',
 'phthalimide_main',
 'pyrrole_main',
 'secondary_amide_main',
 'side_aromatic_bond_fraction',
 'side_chain_volume',
 'side_double_bond_fraction',
 'side_single_bond_fraction',
 'sulfide_main',
 'sulfone_main',
 'tpsa',
 'vinylic_carbon_main',
 'H_main',
 'H_side',
 'distance_between_polymerization_sites'],
 'Melting temperature': ['C_main_arc_volume',
 'C_main_VdW_volume',
 'C_side_arc_volume',
 'C_side_VdW_volume',
 'F_side_dipole_volume',
 'FractionCSP3',
 'H_main_h_donor_volume',
 'H_main_VdW_volume',
 'H_side_h_donor_volume',
 'H_side_VdW_volume',
 'N_main_dipole_volume',
 'N_main_arhet_volume',
 'N_side_dipole_volume',
 'N_side_arhet_volume',
 'NumAliphaticCarbocycles',
 'NumAliphaticHeterocycles',
 'NumAliphaticRings',
 'NumAromaticCarbocycles',
 'NumAromaticHeterocycles',
 'NumHeteroatoms',
 'NumRotatableBonds',
 'NumSaturatedCarbocycles',
 'NumSaturatedHeterocycles',
 'NumSaturatedRings',
 'O_main_dipole_volume',
 'O_side_dipole_volume',
 'O_side_h_donor_volume',
 'RingCount',
 'S_main_dipole_volume',
 'S_main_arhet_volume',
 'Si_main_dipole_volume',
 'alyphatic_carbon_main',
 'alyphatic_carbon_side',
 'aniline_main',
 'arc_feature_main',
 'arc_feature_side',
 'arhet_feature_main',
 'arhet_feature_side',
 'balaban_index',
 'benzene_main',
 'benzene_side',
 'carbamate_main',
 'ester_main',
 'ester_side',
 'ether_main',
 'ether_side',
 'rotatable_bond_fraction',
 'h_donor_feature_main',
 'h_donor_feature_side',
 'hallkier_alpha',
 'imide_main',
 'log_p',
 'main_chain_volume',
 'main_double_bond_fraction',
 'mol_mass',
 'dipole_feature_main',
 'dipole_feature_side',
 'phthalimide_main',
 'pyrrole_main',
 'secondary_amide_main',
 'side_aromatic_bond_fraction',
 'side_chain_volume',
 'side_double_bond_fraction',
 'side_single_bond_fraction',
 'sulfide_main',
 'sulfone_main',
 'tpsa',
 'vinylic_carbon_main',
 'H_main',
 'H_side',
 'distance_between_polymerization_sites']}

def get_vdw_radius(atom):
    """Returns the Van der Waals radius of an atom."""
    return vdw_radii.get(atom.GetSymbol(), None)

def calculate_sphere_volume(radius):
    """Calculates the volume of a sphere from its radius."""
    return (4 / 3) * pi * (radius**3)

def calculate_overlap_volume(r1, r2, d):
    """Calculates the volume of a spherical segment cut off by a neighboring atom with radius r2."""
    if d >= r1 + r2:  # The spheres do not overlap
        return 0
    if d <= abs(r1 - r2):  # One sphere is completely inside the other
        return calculate_sphere_volume(min(r1, r2))
    # General case of overlap
    # h - segment hight
    h = r1 - ((r1**2 + d**2 - r2**2) / (2 * d))
    V = (1/3) * pi * (h**2) * ((3 * r1) - h)
    return V

def find_matches(mol, smarts_list):
    """Generates a list of unique atoms included in the matched SMARTS"""
    if mol is None:
        return None  
    
    unique_indexes = set()  # Use a set for unique indices

    for smarts in smarts_list:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:  
                unique_indexes.update(match)  

    return list(unique_indexes)  

def divide_atoms(original_mol):
    """Splits the molecule’s atoms into 6 index lists based on main/side chain and VdW / Hydrogen bond / Dipole-dipole interactions"""
    try:
        # Step 1: Identify the polymerization atoms (*)
        original_mol = Chem.AddHs(original_mol)
        polymerization_atoms = [atom.GetIdx() for atom in original_mol.GetAtoms() if atom.GetSymbol() == '*']
        if len(polymerization_atoms) != 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan  # Return NaN if the number of * symbols is not equal to 2

        # Step 2: Generate a list of main-chain and side-chain atoms
        main_chain_atoms = set(Chem.rdmolops.GetShortestPath(original_mol, polymerization_atoms[0], polymerization_atoms[1]))
        
        # Add all covalently bonded atoms to main_chain_atoms
        for atom_idx in list(main_chain_atoms):  
            atom = original_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                main_chain_atoms.add(neighbor_idx)  

        # Creation of the side-chain atom list
        all_atoms = set(range(original_mol.GetNumAtoms()))  
        side_chain_atoms = all_atoms - main_chain_atoms  

        # Step 3: Split the lists into cations, anions, H-donors, dipolar, and VdW atoms

        METAL_SMARTS = [
                        "[Li]", "[Be]", "[Na]", "[Mg]", "[Al]", "[K]", "[Ca]", "[Sc]", "[Ti]", "[V]", "[Cr]", "[Mn]",
                        "[Fe]", "[Co]", "[Ni]", "[Cu]", "[Zn]", "[Ga]", "[Rb]", "[Sr]", "[Y]", "[Zr]", "[Nb]", "[Mo]",
                        "[Tc]", "[Ru]", "[Rh]", "[Pd]", "[Ag]", "[Cd]", "[In]", "[Sn]", "[Sb]", "[Cs]", "[Ba]", "[La]",
                        "[Ce]", "[Pr]", "[Nd]", "[Pm]", "[Sm]", "[Eu]", "[Gd]", "[Tb]", "[Dy]", "[Ho]", "[Er]", "[Tm]",
                        "[Yb]", "[Lu]", "[Hf]", "[Ta]", "[W]", "[Re]", "[Os]", "[Ir]", "[Pt]", "[Au]", "[Hg]", "[Tl]",
                        "[Pb]", "[Bi]", "[Po]", "[Fr]", "[Ra]", "[Ac]", "[Th]", "[Pa]", "[U]", "[Np]", "[Pu]", "[Am]",
                        "[Cm]", "[Bk]", "[Cf]", "[Es]", "[Fm]", "[Md]", "[No]", "[Lr]"
                    ]

        cation_atoms = set(find_matches(original_mol, METAL_SMARTS))

        anion_atoms = set()
        for atom_idx in cation_atoms:
            atom = original_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                anion_atoms.add(neighbor.GetIdx())

        cation_atoms -= anion_atoms
        
        DONOR_SMARTS = ["[OH]", "[NH]", "[NH2]", "[PH]", "[SH]", "[SiH]", "[nH]"]
        AROMATIC_HETEROATOMS_SMARTS = ["[nH0]", "[o]", "[s]"]
        dipole_SMARTS = ["[O]", "[N]", "[S]", "[P]", "[Si]", "[Cl]", "[Br]", "[F]"]

        h_donor_atoms = set(find_matches(original_mol, DONOR_SMARTS))

        # Add hydrogens bonded to H-donor atoms
        h_donor_hydrogens = set()
        for atom_idx in h_donor_atoms:
            atom = original_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == "H":
                    h_donor_hydrogens.add(neighbor.GetIdx())
        
        # Merge the indices of donor atoms and their hydrogens
        h_donor_atoms.update(h_donor_hydrogens)
        h_donor_atoms -= cation_atoms
        h_donor_atoms -= anion_atoms

        # Search for aromatic heteroatoms
        aromatic_heteroatoms = set(find_matches(original_mol, AROMATIC_HETEROATOMS_SMARTS))
        aromatic_heteroatoms -= h_donor_atoms
        aromatic_heteroatoms -= cation_atoms
        aromatic_heteroatoms -= anion_atoms

        dipole_atoms = set(find_matches(original_mol, dipole_SMARTS))

        # Remove from the dipolar list the atoms that belong to the H-donor list
        dipole_atoms -= h_donor_atoms
        dipole_atoms -= cation_atoms
        dipole_atoms -= anion_atoms
        dipole_atoms -= aromatic_heteroatoms

        # Search for aromatic carbon atoms
        AROMATIC_CARBON_SMARTS = ["[c]"]
        aromatic_carbons = set(find_matches(original_mol, AROMATIC_CARBON_SMARTS))
        aromatic_carbons -= dipole_atoms
        aromatic_carbons -= h_donor_atoms
        aromatic_carbons -= cation_atoms
        aromatic_carbons -= anion_atoms
        aromatic_carbons -= aromatic_heteroatoms

        # Atoms belonging to VdW intermolecular interactions
        VdW_atoms = all_atoms - h_donor_atoms - dipole_atoms - cation_atoms - anion_atoms - aromatic_heteroatoms - aromatic_carbons

        # Dividing main_chain_atoms
        main_chain_VdW_atoms = main_chain_atoms & VdW_atoms
        main_chain_h_donor_atoms = main_chain_atoms & h_donor_atoms
        main_chain_dipole_atoms = main_chain_atoms & dipole_atoms
        main_chain_cation_atoms = main_chain_atoms & cation_atoms
        main_chain_anion_atoms = main_chain_atoms & anion_atoms
        main_chain_arhet_atoms = main_chain_atoms & aromatic_heteroatoms
        main_chain_arc_atoms = main_chain_atoms & aromatic_carbons

        # Dividing side_chain_atoms
        side_chain_VdW_atoms = side_chain_atoms & VdW_atoms
        side_chain_h_donor_atoms = side_chain_atoms & h_donor_atoms
        side_chain_dipole_atoms = side_chain_atoms & dipole_atoms
        side_chain_cation_atoms = side_chain_atoms & cation_atoms
        side_chain_anion_atoms = side_chain_atoms & anion_atoms
        side_chain_arhet_atoms = side_chain_atoms & aromatic_heteroatoms
        side_chain_arc_atoms = side_chain_atoms & aromatic_carbons

        # Return 14 lists
        return (list(main_chain_VdW_atoms), list(main_chain_h_donor_atoms), list(main_chain_dipole_atoms), list(main_chain_cation_atoms),
                list(main_chain_anion_atoms), list(main_chain_arhet_atoms), list(main_chain_arc_atoms),
                list(side_chain_VdW_atoms), list(side_chain_h_donor_atoms), list(side_chain_dipole_atoms), list(side_chain_cation_atoms),
                list(side_chain_anion_atoms), list(side_chain_arhet_atoms), list(side_chain_arc_atoms))

    except:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan


def calculate_volumes(original_mol):
    try:
        if original_mol is None:
            return (np.nan,) * 16

        original_mol = Chem.AddHs(original_mol)

        main_VdW, main_h_donor, main_dipole, main_cation, main_anion, main_arhet, main_arc, \
        side_VdW, side_h_donor, side_dipole, side_cation, side_anion, side_arhet, side_arc = divide_atoms(original_mol)

        for atom in original_mol.GetAtoms():
            if atom.GetSymbol() == "*":
                atom.SetProp("is_wildcard", "true")
                atom.SetAtomicNum(1)

        mol = Chem.Mol(original_mol)

        try:
            result = AllChem.EmbedMolecule(mol, maxAttempts=10, randomSeed=42)
        except Exception as e:
            print(f"⚠️ Error EmbedMolecule: {e}")

        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            print(f"⚠️ Error UFFOptimizeMolecule: {e}")

        distance_between_polymerization_sites = None
        try:
            conf = mol.GetConformer()
            atom_coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
            wildcard_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.HasProp("is_wildcard")]
            if len(wildcard_indices) == 2:
                i1, i2 = wildcard_indices
                distance_between_polymerization_sites = np.linalg.norm(atom_coords[i1] - atom_coords[i2])
        except Exception as e:
            atom_coords = None

        # Fallback: if no 3D conformation is available, use 2D coordinates
        if distance_between_polymerization_sites is None:
            try:
                mol2D = Chem.Mol(mol)
                AllChem.Compute2DCoords(mol2D)
                conf2d = mol2D.GetConformer()
                atom_coords_2d = np.array([conf2d.GetAtomPosition(i) for i in range(mol2D.GetNumAtoms())])
                wildcard_indices = [atom.GetIdx() for atom in mol2D.GetAtoms() if atom.HasProp("is_wildcard")]
                if len(wildcard_indices) == 2:
                    i1, i2 = wildcard_indices
                    distance_between_polymerization_sites = np.linalg.norm(atom_coords_2d[i1] - atom_coords_2d[i2])
                else:
                    print("❌ 2 atoms with is_wildcard flag not found in 2D")
            except Exception as e:
                print(f"💥 Error while calculating 2D-distance: {e}")

        for atom in mol.GetAtoms():
            if atom.HasProp("is_wildcard"):
                atom.SetAtomicNum(0)
                atom.ClearProp("is_wildcard")

        volumes = {k: [] for k in [
            "main_VdW", "main_h_donor", "main_dipole", "main_cation", "main_anion", "main_arhet", "main_arc",
            "side_VdW", "side_h_donor", "side_dipole", "side_cation", "side_anion", "side_arhet", "side_arc"
        ]}
        total_corrected_volume = 0

        for i, atom in enumerate(mol.GetAtoms()):
            r1 = get_vdw_radius(atom)
            if not r1:
                continue

            volume = calculate_sphere_volume(r1)
            overlaps = []
            for bond in atom.GetBonds():
                neighbor_idx = bond.GetOtherAtomIdx(i)
                neighbor = mol.GetAtomWithIdx(neighbor_idx)
                r2 = get_vdw_radius(neighbor)
                if not r2:
                    continue
                if atom_coords is not None:
                    d = np.linalg.norm(atom_coords[i] - atom_coords[neighbor_idx])
                else:
                    d = 1.5 * (r1 + r2)
                overlaps.append(calculate_overlap_volume(r1, r2, d))

            for overlap in overlaps:
                volume -= overlap
            volume = max(volume, 0)

            atom_symbol = atom.GetSymbol()
            atom_data = (atom_symbol, volume)

            if i in main_VdW:
                volumes["main_VdW"].append(atom_data)
            elif i in main_h_donor:
                volumes["main_h_donor"].append(atom_data)
            elif i in main_dipole:
                volumes["main_dipole"].append(atom_data)
            elif i in main_cation:
                volumes["main_cation"].append(atom_data)
            elif i in main_anion:
                volumes["main_anion"].append(atom_data)
            elif i in main_arhet:
                volumes["main_arhet"].append(atom_data)
            elif i in main_arc:
                volumes["main_arc"].append(atom_data)
            elif i in side_VdW:
                volumes["side_VdW"].append(atom_data)
            elif i in side_h_donor:
                volumes["side_h_donor"].append(atom_data)
            elif i in side_dipole:
                volumes["side_dipole"].append(atom_data)
            elif i in side_cation:
                volumes["side_cation"].append(atom_data)
            elif i in side_anion:
                volumes["side_anion"].append(atom_data)
            elif i in side_arhet:
                volumes["side_arhet"].append(atom_data)
            elif i in side_arc:
                volumes["side_arc"].append(atom_data)

            total_corrected_volume += volume

        return (
            list(volumes["main_VdW"]),
            list(volumes["main_h_donor"]),
            list(volumes["main_dipole"]),
            list(volumes["main_cation"]),
            list(volumes["main_anion"]),
            list(volumes["main_arhet"]),
            list(volumes["main_arc"]),
            list(volumes["side_VdW"]),
            list(volumes["side_h_donor"]),
            list(volumes["side_dipole"]),
            list(volumes["side_cation"]),
            list(volumes["side_anion"]),
            list(volumes["side_arhet"]),
            list(volumes["side_arc"]),
            total_corrected_volume,
            distance_between_polymerization_sites
        )

    except:
        return (np.nan,) * 16






# Function to calculate the sum of volumes for each unique element
def summarize_volumes(atom_volume_list):
    volume_dict = {}
    for atom, volume in atom_volume_list:
        volume_dict[atom] = volume_dict.get(atom, 0) + volume
    return volume_dict

# Function to output the volume sums by unique atoms
def print_atom_volumes(title, atom_volumes):
    print(title)
    for atom, volume in atom_volumes.items():
        print(f"{atom}: {volume:.2f} Å³")
    print()


def split_chains(original_mol):
    """Splits atoms into two index lists: main chain and side chain"""
    try:
        # Step 1: Find the polymerization atoms (*)
        if original_mol is None:
            return np.nan, np.nan
        
        original_mol = Chem.AddHs(original_mol)
        polymerization_atoms = [atom.GetIdx() for atom in original_mol.GetAtoms() if atom.GetSymbol() == '*']
        if len(polymerization_atoms) != 2:
            return np.nan, np.nan  # Return NaN if there are not exactly 2 * symbols

        # Step 2: Generate the list of main-chain and side-chain atoms
        main_chain_atoms = set(Chem.rdmolops.GetShortestPath(original_mol, polymerization_atoms[0], polymerization_atoms[1]))
        
        # Add all covalently bonded atoms to main_chain_atoms
        for atom_idx in list(main_chain_atoms):  
            atom = original_mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                main_chain_atoms.add(neighbor_idx)  

        # Creation of the side-chain atom list
        all_atoms = set(range(original_mol.GetNumAtoms()))  
        side_chain_atoms = all_atoms - main_chain_atoms  

        return main_chain_atoms, side_chain_atoms

    except:
        return np.nan, np.nan


def count_functional_groups_by_chain(mol, main_chain_atoms, side_chain_atoms, smarts_pattern):
    """
    Counts the occurrences of a functional group in the main and side chains of a molecule.

    Parameters:
    mol (Mol): The molecule in mol representation.
    main_chain_atoms (list): List of atom indices belonging to the main chain.
    side_chain_atoms (list): List of atom indices belonging to the side chain.
    smarts_pattern (str): SMARTS pattern of the target functional group.

    Returns:
    tuple: Number of matches in the main chain, number of matches in the side chain.
    """
    try:

        # Convert the SMARTS pattern into a mol object
        smarts = Chem.MolFromSmarts(smarts_pattern)
        if smarts is None:
            raise ValueError(f"Некорректный SMARTS-паттерн: {smarts_pattern}")

        # Find all matches of the SMARTS pattern
        matches = mol.GetSubstructMatches(smarts)

        # Split the matches into main-chain and side-chain
        main_count = 0
        side_count = 0
        for match in matches:
            # Check which chain contains more atoms of the match
            main_atoms = sum(1 for atom_idx in match if atom_idx in main_chain_atoms)
            side_atoms = sum(1 for atom_idx in match if atom_idx in side_chain_atoms)

            if main_atoms > side_atoms:
                main_count += 1
            elif side_atoms > main_atoms:
                side_count += 1

        return main_count, side_count

    except:
        return np.nan, np.nan

def count_bond_types_by_chain(mol, main_chain_atoms, side_chain_atoms):
    """
    Computes the fractions of single, double, triple, and aromatic bonds
    in the main and side chains of a molecule.

    Parameters:
    mol (Mol): The molecule in mol format.
    main_chain_atoms (list, set, tuple, or str): Indices of atoms belonging to the main chain.
        If provided as a string, it will be converted to a set.
    side_chain_atoms (list, set, tuple, or str): Indices of atoms belonging to the side chain.
        If provided as a string, it will be converted to a set.

    Returns:
    tuple: Bond fractions in the format:
        (main_single_fraction, main_double_fraction, main_triple_fraction, main_aromatic_fraction,
         side_single_fraction, side_double_fraction, side_triple_fraction, side_aromatic_fraction)
        — Fractions of single, double, triple, and aromatic bonds in the main and side chains.

    Notes:
    - Aromatic bonds are counted separately using rdchem.BondType.AROMATIC.
    """
    try:
        # Conversion of strings to sets, if necessary
        if isinstance(main_chain_atoms, str):
            try:
                main_chain_atoms = set(eval(main_chain_atoms))
            except Exception:
                return 0, 0, 0, 0, 0, 0, 0, 0

        if isinstance(side_chain_atoms, str):
            try:
                side_chain_atoms = set(eval(side_chain_atoms))
            except Exception:
                side_chain_atoms = set()

        # Check the validity of atom indices
        atom_count = mol.GetNumAtoms()
        main_chain_atoms = [idx for idx in main_chain_atoms if isinstance(idx, int) and idx < atom_count]
        side_chain_atoms = [idx for idx in side_chain_atoms if isinstance(idx, int) and idx < atom_count]

        if not main_chain_atoms:
            return 0, 0, 0, 0, 0, 0, 0, 0

        # Lists for storing bonds of the main and side chains
        main_bonds = []
        side_bonds = []

        # Iterate over all bonds in the molecule
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type = bond.GetBondType()

            # Check which chain the bond atoms belong to
            in_main = begin_idx in main_chain_atoms or end_idx in main_chain_atoms
            in_side = begin_idx in side_chain_atoms or end_idx in side_chain_atoms

            # Add the bond to the corresponding list
            if in_main and not in_side:
                main_bonds.append(bond_type)
            elif in_side and not in_main:
                side_bonds.append(bond_type)

        # Count the bond types for the main chain
        main_single = main_bonds.count(Chem.rdchem.BondType.SINGLE)
        main_double = main_bonds.count(Chem.rdchem.BondType.DOUBLE)
        main_triple = main_bonds.count(Chem.rdchem.BondType.TRIPLE)
        main_aromatic = main_bonds.count(Chem.rdchem.BondType.AROMATIC)
        main_total = len(main_bonds)

        # Count the bond types for the side chain
        side_single = side_bonds.count(Chem.rdchem.BondType.SINGLE)
        side_double = side_bonds.count(Chem.rdchem.BondType.DOUBLE)
        side_triple = side_bonds.count(Chem.rdchem.BondType.TRIPLE)
        side_aromatic = side_bonds.count(Chem.rdchem.BondType.AROMATIC)
        side_total = len(side_bonds)

        # Compute the bond fractions while avoiding division by zero
        main_single_fraction = main_single / main_total if main_total > 0 else 0
        main_double_fraction = main_double / main_total if main_total > 0 else 0
        main_triple_fraction = main_triple / main_total if main_total > 0 else 0
        main_aromatic_fraction = main_aromatic / main_total if main_total > 0 else 0

        side_single_fraction = side_single / side_total if side_total > 0 else 0
        side_double_fraction = side_double / side_total if side_total > 0 else 0
        side_triple_fraction = side_triple / side_total if side_total > 0 else 0
        side_aromatic_fraction = side_aromatic / side_total if side_total > 0 else 0

        return main_single_fraction, main_double_fraction, main_triple_fraction, main_aromatic_fraction, side_single_fraction, side_double_fraction, side_triple_fraction, side_aromatic_fraction

    except:
        return 0, 0, 0, 0, 0, 0, 0, 0


def calculate_descriptors(mol):
    return {
        'FractionCSP3': Descriptors.FractionCSP3(mol),
        'HeavyAtomCount': Descriptors.HeavyAtomCount(mol),
        'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles(mol),
        'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
        'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles(mol),
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
        'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        'NumHeteroatoms': (Descriptors.NumHeteroatoms(mol) - 2),
        'NumRotatableBonds': (Descriptors.NumRotatableBonds(mol) + 1),
        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles(mol),
        'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
        'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
        'RingCount': Descriptors.RingCount(mol),
        'log_p' : Crippen.MolLogP(mol),
        'tpsa' : rdMolDescriptors.CalcTPSA(mol),
        'mol_mass': Descriptors.MolWt(mol)
    }

def calculate_rotatable_bond_fraction(mol):
    try:
        # Total number of bonds
        total_bonds = mol.GetNumBonds()
        if total_bonds == 0:  
            return 0
        # Number of rotatable bonds
        rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol) + 1
        # Fraction of rotatable bonds
        return rotatable_bonds / total_bonds
    except:
        return None

# Function to calculate the Balaban index
def calculate_balaban_index(mol):
    try:
        balaban_index = GraphDescriptors.BalabanJ(mol)
        return balaban_index
    except:
        return None

# Function to calculate the structural complexity index HallKierAlpha
def calculate_hallkier_alpha(mol):
    try:
        hallkier_alpha = Descriptors.HallKierAlpha(mol)
        return hallkier_alpha
    except:
        return None

def smiles_to_xyz_xtb(smiles: str, random_seed: int = 42) -> str:
    """Generates an XYZ file for the molecule, replacing * with [H]."""
    smiles = smiles.replace("*", "[H]")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # use provided random seed
    if AllChem.EmbedMolecule(mol, randomSeed=random_seed) != 0:
        return None

    try:
        AllChem.UFFOptimizeMolecule(mol)
    except:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            return None

    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    if any(math.isnan(pos.x) or math.isnan(pos.y) or math.isnan(pos.z) for pos in coords):
        return None
    lines = [f"{len(atoms)}", ""]
    for atom, pos in zip(atoms, coords):
        lines.append(f"{atom} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}")
    return "\n".join(lines)



def run_xtb(smiles: str, max_attempts: int = 3) -> dict:
    """Runs xTB and extracts energy, charge, and dipole descriptors."""
    for attempt in range(max_attempts):
        # new: assign unique seed per attempt
        seed = 42 + attempt
        xyz_data = smiles_to_xyz_xtb(smiles, random_seed=seed)
        if xyz_data is None:
            continue

        with tempfile.TemporaryDirectory() as tmpdir:
            xyz_path = os.path.join(tmpdir, "mol.xyz")
            with open(xyz_path, "w") as f:
                f.write(xyz_data)

            try:
                result = subprocess.run(
                    ["xtb", "mol.xyz", "--gfn", "2", "--opt", "normal", "--parallel", "10"],
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1800
                )
            except subprocess.TimeoutExpired:
                print(f"[xTB ERROR] Timeout for SMILES: {smiles} (attempt {attempt+1})")
                continue
            except Exception as e:
                print(f"[xTB ERROR] Launch error xTB: {e} (attempt {attempt+1})")
                continue

            if result.returncode != 0:
                print(f"[xTB] Attempt {attempt+1}: error ({result.returncode})\n{result.stderr}")
                continue

            output = result.stdout

            def extract_last(pattern: str):
                matches = re.findall(pattern, output)
                return float(matches[-1]) if matches else None

            desc = {
                "total_energy (hartree)": extract_last(r"TOTAL\s+ENERGY\s+(-?\d+\.\d+)"),
                "homo_lumo_gap (eV)": extract_last(r"HOMO.*?LUMO\s+GAP\s+(-?\d+\.\d+)"),
                "homo_energy (eV)": extract_last(r"(-?\d+\.\d+)\s+\(HOMO\)"),
                "lumo_energy (eV)": extract_last(r"(-?\d+\.\d+)\s+\(LUMO\)")
            }

            # Extract the dipole moment from the "full:" block
            dipole_moment = None
            for line in output.splitlines():
                if line.strip().startswith("full:"):
                    parts = line.strip().split()
                    if len(parts) == 5:
                        try:
                            dipole_moment = float(parts[-1])
                        except:
                            pass
                    break
            desc["dipole_moment (D)"] = dipole_moment

            # Parse Mulliken charges from the charges file
            charges_path = os.path.join(tmpdir, "charges")
            if os.path.exists(charges_path):
                try:
                    with open(charges_path, "r") as f:
                        charges = [float(line.strip()) for line in f if line.strip()]
                    charges_np = np.array(charges)
                    desc.update({
                        "mulliken_charge_std (e)": charges_np.std(),
                        "mulliken_charge_min (e)": charges_np.min(),
                        "mulliken_charge_max (e)": charges_np.max()
                    })
                except Exception as e:
                    print(f"[xTB] Error while reading Mulliken charges: {e}")
            else:
                print("[xTB] File charges not found — Mulliken charges were not extracted.")

            return desc

    print(f"[xTB FAIL] All {max_attempts} attempts were not successful for: {smiles}")
    return {}


def featurize(smiles, with_xtb: bool = True):
    
    """
    Function to compute all features for a SMILES string.
    Returns a dictionary of feature : value.

    Parameters
    ----------
    smiles : str
        Input SMILES string with exactly two '*' atoms.
    with_xtb : bool, optional (default=True)
        If True, compute xTB descriptors. 
        If False, skip xTB and set their values to None.

    Returns
    -------
    dict
        Dictionary of features with the same length and order as all_chosen_features.
    """

    xtb_feature_names = [
        "total_energy (hartree)",
        "homo_lumo_gap (eV)",
        "homo_energy (eV)",
        "lumo_energy (eV)",
        "dipole_moment (D)",
        "mulliken_charge_std (e)",
        "mulliken_charge_min (e)",
        "mulliken_charge_max (e)",
    ]

    final_dict = dict()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    if smiles.count('*') != 2:
        raise ValueError(f"SMILES must contain exactly 2 '*' atoms, got {smiles.count('*')} in {smiles}")
    
    for feature in all_chosen_features:
        final_dict[feature] = 0

    calculated = calculate_volumes(mol)
    volumes_dict = dict(zip(volumes, calculated[:-1]))  
    distance_between_polymerization_sites = calculated[-1]
    final_dict["distance_between_polymerization_sites"] = distance_between_polymerization_sites


    # Collect unique elements across all volume_col
    unique_elements = set()
    for col in atom_volumes_list:
        values = volumes_dict.get(col, [])
        for d in values:
            if d is not None and len(d) == 2:
                el, v = d
                if v > 0:
                    unique_elements.add(el)

    # Aggregate by each element and column
    for element in unique_elements:
        for col in atom_volumes_list:
            if col not in volumes_dict:
                continue
            key = f"{element}_{col.replace('_atoms_volumes', '')}_volume"
            total_vol = sum(
                d[1] for d in volumes_dict[col]
                if d is not None and len(d) == 2 and d[0] == element
            )
            final_dict[key] = total_vol


    for volume_type in volume_types:
        final_dict[f'sum_of_{volume_type}_volumes_main'] = sum(final_dict[col] for col in final_dict if f'main_{volume_type}_volume' in col)
        final_dict[f'sum_of_{volume_type}_volumes_side'] = sum(final_dict[col] for col in final_dict if f'side_{volume_type}_volume' in col)

    final_dict['main_chain_volume'] = sum(final_dict[f'sum_of_{volume_type}_volumes_main'] for volume_type in volume_types)
    final_dict['side_chain_volume'] = sum(final_dict[f'sum_of_{volume_type}_volumes_side'] for volume_type in volume_types)

    for i in range(len(volume_types)):
        for chain in ['main', 'side']:
            if final_dict[f'{chain}_chain_volume'] != 0:
                final_dict[f'{volume_types[i]}_feature_{chain}'] = (
                    final_dict[f'sum_of_{volume_types[i]}_volumes_{chain}'] /
                    final_dict[f'{chain}_chain_volume']
                )
            else:
                final_dict[f'{volume_types[i]}_feature_{chain}'] = 0

    # Calculation of side and main chains using SMARTS
    mol_with_H = Chem.AddHs(mol)
    chain_split = split_chains(mol_with_H)

    atom_counts_main = {'F': 0, 'Cl': 0, 'Br': 0, 'H': 0}
    atom_counts_side = {'F': 0, 'Cl': 0, 'Br': 0, 'H': 0}
        
    for atom in mol_with_H.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_counts_main:
            idx = atom.GetIdx()
            if idx in chain_split[0]:
                atom_counts_main[symbol] += 1
            elif idx in chain_split[1]:
                atom_counts_side[symbol] += 1
        
    for el in atom_counts_main.keys():
        final_dict[f"{el}_main"] = atom_counts_main[el]
        final_dict[f"{el}_side"] = atom_counts_side[el]

    for key in ALLSMARTS.keys():
        final_dict[f'{ALLSMARTS[key]}_main'], final_dict[f'{ALLSMARTS[key]}_side'] = count_functional_groups_by_chain(mol, chain_split[0], chain_split[1], key)

    # If phosphoric acid is present, subtract its contribution from po3_derivative
    for chain in ['main', 'side']:
        acid_key = f'phosphoric_acid_{chain}'
        po3_key = f'po3_derivative_{chain}'
        if acid_key in final_dict and po3_key in final_dict:
            correction = 3 * final_dict[acid_key]
            final_dict[po3_key] = max(0, final_dict[po3_key] - correction)


    (final_dict['main_single_bond_fraction'], final_dict['main_double_bond_fraction'], final_dict['main_triple_bond_fraction'],
    final_dict['main_aromatic_bond_fraction'], final_dict['side_single_bond_fraction'], final_dict['side_double_bond_fraction'],
    final_dict['side_triple_bond_fraction'], final_dict['side_aromatic_bond_fraction']) = count_bond_types_by_chain(mol, chain_split[0], chain_split[1])

    a = calculate_descriptors(mol)
    for key in a.keys():
        final_dict[key] = a[key]

    final_dict['rotatable_bond_fraction'] = calculate_rotatable_bond_fraction(mol)
    final_dict['balaban_index'] = calculate_balaban_index(mol)
    final_dict['hallkier_alpha'] = calculate_hallkier_alpha(mol)

    # Calculation of xTB descriptors
    if with_xtb:
        xtb_results = run_xtb(smiles)
        final_dict["total_energy (hartree)"] = xtb_results.get("total_energy (hartree)", None)
        final_dict["homo_lumo_gap (eV)"] = xtb_results.get("homo_lumo_gap (eV)", None)
        final_dict["homo_energy (eV)"] = xtb_results.get("homo_energy (eV)", None)
        final_dict["lumo_energy (eV)"] = xtb_results.get("lumo_energy (eV)", None)
        final_dict["dipole_moment (D)"] = xtb_results.get("dipole_moment (D)", None)
        final_dict["mulliken_charge_std (e)"] = xtb_results.get("mulliken_charge_std (e)", None)
        final_dict["mulliken_charge_min (e)"] = xtb_results.get("mulliken_charge_min (e)", None)
        final_dict["mulliken_charge_max (e)"] = xtb_results.get("mulliken_charge_max (e)", None)
    else:
        for k in xtb_feature_names:
            if k in all_chosen_features:
                final_dict[k] = None

    output = { feature: final_dict[feature] for feature in all_chosen_features }
    return output

def rename_features(names, features):
    return {new_name: features.get(old_name, None) for old_name, new_name in zip(all_chosen_features, names)}

def featurize_to_rus_dict(SMILES, with_xtb: bool = True):
    """
    Function to compute all features in Russian for a SMILES string. 
    Returns a dictionary of feature : value.
    
    Parameters
    ----------
    SMILES : str
        Input SMILES string with exactly two '*' atoms.
    with_xtb : bool, optional (default=True)
        If True, compute xTB descriptors. If False, skip xTB and set their values to None.
    """
    return rename_features(all_chosen_features_rus, featurize(SMILES, with_xtb=with_xtb))

def featurize_to_eng_dict(SMILES, with_xtb: bool = True):
    """
    Function to compute all features in English for a SMILES string. 
    Returns a dictionary of feature : value.
    
    Parameters
    ----------
    SMILES : str
        Input SMILES string with exactly two '*' atoms.
    with_xtb : bool, optional (default=True)
        If True, compute xTB descriptors. If False, skip xTB and set their values to None.
    """
    return rename_features(all_chosen_features_eng, featurize(SMILES, with_xtb=with_xtb))

# ==========================
# Prediction utilities
# ==========================

def _build_X_for_target(full_feat_dict: dict, feature_list: list) -> pd.DataFrame:
    """
    Builds a DataFrame with a single row for a specific target.
    The column order matches feature_list. Missing features -> NaN.
    """
    row = {f: full_feat_dict.get(f, np.nan) for f in feature_list}
    return pd.DataFrame([row], columns=feature_list)


def _predict_mean_std(model, X_df: pd.DataFrame):
    """
    Calls predict for a GPR model from autogpr.
    Expected that model.predict(TabularData.from_pandas(X=...))
    returns an object with .y and .y_std.
    """
    X_tab = TabularData.from_pandas(X=X_df)
    preds = model.predict(X_tab)
    mean = float(preds.y[0])       # mean predicted value
    std  = float(preds.y_std[0])   # prediction standard deviation
    return mean, std


# ==========================
# Model loading utilities
# ==========================

# Mapping: model name in globals() -> filename
_MODEL_FILES = {
    "Density": "Density.pkl",
    "Glass transition temperature": "Glass transition temperature.pkl",
    "Thermal decomposition temperature": "Thermal decomposition temperature.pkl",
    "Melting temperature": "Melting temperature.pkl",

    "Density_no_xtb": "Density_no_xtb.pkl",
    "Glass transition temperature_no_xtb": "Glass transition temperature_no_xtb.pkl",
    "Thermal decomposition temperature_no_xtb": "Thermal decomposition temperature_no_xtb.pkl",
    "Melting temperature_no_xtb": "Melting temperature_no_xtb.pkl",
}


def load_models_from_dir(model_dir: str | os.PathLike) -> dict:
    """
    Loads available .pkl models from model_dir and registers them in globals()
    under the expected names (keys of _MODEL_FILES).
    Returns a dict {model_name: model_object} of actually loaded models.
    """
    model_dir = Path(model_dir)
    loaded = {}
    for model_name, file_name in _MODEL_FILES.items():
        path = model_dir / file_name
        if path.exists():
            try:
                model_obj = joblib.load(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load '{file_name}': {e}")
            globals()[model_name] = model_obj
            loaded[model_name] = model_obj
    return loaded


def _ensure_models_loaded(with_xtb: bool, search_dir: str | os.PathLike | None = None):
    """
    Ensures that required models for the current with_xtb flag are present in globals().
    If not, tries to load them from search_dir (defaults to the directory of this file).
    """
    required = [
        "Density",
        "Glass transition temperature",
        "Thermal decomposition temperature",
        "Melting temperature",
    ]
    if not with_xtb:
        required = [f"{r}_no_xtb" for r in required]

    missing = [m for m in required if m not in globals()]
    if missing:
        base = Path(search_dir) if search_dir is not None else Path(__file__).resolve().parent
        load_models_from_dir(base)
        still_missing = [m for m in required if m not in globals()]
        if still_missing:
            raise KeyError(f"Models not found after loading: {', '.join(still_missing)}")

# xTB feature names used to decide fallback
_XTB_FEATURES = [
    "total_energy (hartree)",
    "homo_lumo_gap (eV)",
    "homo_energy (eV)",
    "lumo_energy (eV)",
    "dipole_moment (D)",
    "mulliken_charge_std (e)",
    "mulliken_charge_min (e)",
    "mulliken_charge_max (e)",
]

def _has_missing_xtb(feat_dict: dict) -> bool:
    """Return True if any xTB descriptor is missing (None or NaN)."""
    for k in _XTB_FEATURES:
        v = feat_dict.get(k, None)
        # считаем пропуском и None, и NaN
        if v is None:
            return True
        try:
            if pd.isna(v):
                return True
        except Exception:
            pass
    return False


def predict_targets(
    smiles: str,
    with_xtb: bool = True,
    target: Optional[str] = None
) -> Union[pd.DataFrame, dict]:
    """
    Prediction workflow:
      1) featurization (with/without xTB),
      2) feature selection (features_by_target or *_no_xtb),
      3) run through corresponding model(s),
      4) return:
         - DataFrame with rows per target (default, when target is None),
         - dict {"mean": float, "std": float} for a specific target.

    Fallback: If with_xtb=True but xTB descriptors contain NaN/None,
    automatically switch to no-xTB models.

    Parameters
    ----------
    smiles : str
        Input SMILES with exactly two '*' atoms.
    with_xtb : bool, default True
        Whether to use xTB-based features if available.
    target : Optional[str], default None
        If provided, predict only this target. Otherwise predict all.

    Returns
    -------
    Union[pd.DataFrame, dict]
        - DataFrame with index "target" and columns ["mean", "std"] if target is None
        - dict {"mean": float, "std": float} if a specific target is requested
    """
    # 1) featurization
    feat_dict = featurize(smiles, with_xtb=with_xtb)

    # 1.5) fallback to no-xTB if xTB features are missing
    use_xtb = with_xtb
    if with_xtb and _has_missing_xtb(feat_dict):
        logging.warning("xTB features are missing (NaN/None). Falling back to no-xTB models.")
        use_xtb = False

    # 2) feature map (respecting fallback)
    features_map = features_by_target if use_xtb else features_by_target_no_xtb

    # 2.5) ensure models are loaded
    _ensure_models_loaded(with_xtb=use_xtb)

    # If a specific target is requested
    if target is not None:
        if target not in features_map:
            available = ", ".join(sorted(features_map.keys()))
            raise KeyError(f"Unknown target '{target}'. Available: {available}")

        model_name = target if use_xtb else f"{target}_no_xtb"
        if model_name not in globals():
            raise KeyError(f"Model '{model_name}' not found in the environment")

        model = globals()[model_name]
        X_df = _build_X_for_target(feat_dict, features_map[target])
        mean, std = _predict_mean_std(model, X_df)
        return {"mean": mean, "std": std}

    # Otherwise predict all targets
    results = []
    for tgt, feature_list in features_map.items():
        model_name = tgt if use_xtb else f"{tgt}_no_xtb"
        if model_name not in globals():
            raise KeyError(f"Model '{model_name}' not found in the environment")
        model = globals()[model_name]

        X_df = _build_X_for_target(feat_dict, feature_list)
        mean, std = _predict_mean_std(model, X_df)
        results.append((tgt, mean, std))

    return pd.DataFrame(results, columns=["target", "mean", "std"]).set_index("target")
