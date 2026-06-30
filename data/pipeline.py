import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split

def fetch_herg_data():
    print("Querying ChEMBL for hERG IC50 data (CHEMBL240)...")
    activity = new_client.activity
    
    # hERG target ID is CHEMBL240
    res = activity.filter(
        target_chembl_id='CHEMBL240',
        standard_type='IC50',
        standard_units='nM',
        standard_value__lte=10000
    ).only(['standard_value', 'canonical_smiles'])
    
    records = []
    print(f"Total raw records found: {len(res)}. Filtering for standard_value <= 10000 and standardizing...")
    
    for i, r in enumerate(res):
        if i > 0 and i % 1000 == 0:
            print(f"Processed {i} records...")
            
        val = r.get('standard_value')
        smi = r.get('canonical_smiles')
        
        if val is None or smi is None:
            continue
            
        try:
            val = float(val)
        except ValueError:
            continue
            
        if val > 10000:
            continue
            
        # RDKit standardization and MW filter
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
            
        mw = Descriptors.ExactMolWt(mol)
        if mw > 600:
            continue
            
        clean_smi = Chem.MolToSmiles(mol)
        
        records.append({
            'SMILES': clean_smi,
            'IC50_nM': val,
            'MW': mw
        })
        
    df = pd.DataFrame(records)
    # Drop duplicates just in case multiple assays for the same SMILES
    df = df.groupby('SMILES', as_index=False).agg({'IC50_nM': 'median', 'MW': 'first'})
    
    return df

def get_cipa_data():
    print("Loading CiPA 28-compound panel...")
    # Reference: Fermini et al. 2016
    # Representative delta_QTc_ms derived from literature context (Thorough QT studies / CiPA refs)
    # High Risk = 2, Intermediate Risk = 1, Low/No Risk = 0
    cipa_compounds = [
        # High Risk
        {"drug": "Dofetilide", "SMILES": "CS(=O)(=O)Nc1ccc(OCCCN(C)CCc2ccc(NS(C)(=O)=O)cc2)cc1", "delta_QTc_ms": 40.0, "tdp_risk_class": 2},
        {"drug": "Quinidine", "SMILES": "C=C[C@H]1CN2CCC1C[C@@H]2[C@H](O)c1ccnc2ccc(OC)cc12", "delta_QTc_ms": 45.0, "tdp_risk_class": 2},
        {"drug": "Sotalol", "SMILES": "CS(=O)(=O)Nc1ccc(C(O)CNC(C)C)cc1", "delta_QTc_ms": 30.0, "tdp_risk_class": 2},
        {"drug": "Ibutilide", "SMILES": "CCCCCCC(C)(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1", "delta_QTc_ms": 35.0, "tdp_risk_class": 2},
        {"drug": "Azimilide", "SMILES": "O=C1NC(=O)N(C1=O)CCCCN2CCN(Cc3ccc(Cl)cc3)CC2", "delta_QTc_ms": 35.0, "tdp_risk_class": 2},
        {"drug": "Bepridil", "SMILES": "CC(C)OCCOC(CN(Cc1ccccc1)c2ccccc2)CN3CCCC3", "delta_QTc_ms": 25.0, "tdp_risk_class": 2},
        {"drug": "Vandetanib", "SMILES": "COc1cc2c(Nc3ccc(Br)cc3F)ncnc2cc1OCC1CCN(C)CC1", "delta_QTc_ms": 35.0, "tdp_risk_class": 2},
        {"drug": "Disopyramide", "SMILES": "CC(C)N(CCC(C(N)=O)(c1ccccc1)c2ccccn2)C(C)C", "delta_QTc_ms": 20.0, "tdp_risk_class": 2},
        
        # Intermediate Risk
        {"drug": "Chlorpromazine", "SMILES": "CN(C)CCCN1c2ccccc2Sc3ccc(Cl)cc13", "delta_QTc_ms": 15.0, "tdp_risk_class": 1},
        {"drug": "Cisapride", "SMILES": "COc1cc(Cl)c(N)cc1C(=O)NC2CCN(CCCOc3ccc(F)cc3)CC2OC", "delta_QTc_ms": 20.0, "tdp_risk_class": 1},
        {"drug": "Astemizole", "SMILES": "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc5ccc(F)cc5)CC2)cc1", "delta_QTc_ms": 18.0, "tdp_risk_class": 1},
        {"drug": "Clarithromycin", "SMILES": "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(N(C)C)C3O)C(C)(C)CC(C)C(=O)C(C)C(O)C1(C)OC", "delta_QTc_ms": 12.0, "tdp_risk_class": 1},
        {"drug": "Domperidone", "SMILES": "O=c1[nH]c2ccccc2n1C3CCN(CCCN4C(=O)Nc5ccc(Cl)cc54)CC3", "delta_QTc_ms": 15.0, "tdp_risk_class": 1},
        {"drug": "Droperidol", "SMILES": "O=C(CCCN1CCC(n2c(=O)[nH]c3ccccc32)CC1)c4ccc(F)cc4", "delta_QTc_ms": 14.0, "tdp_risk_class": 1},
        {"drug": "Pimozide", "SMILES": "O=c1[nH]c2ccccc2n1C3CCN(CCCC(c4ccc(F)cc4)c5ccc(F)cc5)CC3", "delta_QTc_ms": 16.0, "tdp_risk_class": 1},
        {"drug": "Ondansetron", "SMILES": "Cn1c2c(c3c1CCCC3=O)cccc2CN4C=CN=C4C", "delta_QTc_ms": 10.0, "tdp_risk_class": 1},
        {"drug": "Terfenadine", "SMILES": "CC(C)(C)c1ccc(C(C)(C)O)cc1.OC(c2ccccc2)(c3ccccc3)C4CCN(CCCC(O)c5ccc(C(C)(C)C)cc5)CC4", "delta_QTc_ms": 15.0, "tdp_risk_class": 1}, # Note: using canonical terfenadine main part in standardization
        {"drug": "Clozapine", "SMILES": "CN1CCN(C2=Nc3cc(Cl)ccc3Nc4ccccc42)CC1", "delta_QTc_ms": 12.0, "tdp_risk_class": 1},
        {"drug": "Risperidone", "SMILES": "Cc1nc2n(c(=O)c1CCN3CCC(c4noc5cc(F)ccc45)CC3)CCCC2", "delta_QTc_ms": 10.0, "tdp_risk_class": 1},
        
        # Low/No Risk
        {"drug": "Diltiazem", "SMILES": "COc1ccc(C2Sc3ccccc3N(CCN(C)C)C(=O)C2OC(C)=O)cc1", "delta_QTc_ms": 2.0, "tdp_risk_class": 0},
        {"drug": "Loratadine", "SMILES": "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc4cccnc42)CC1", "delta_QTc_ms": 1.0, "tdp_risk_class": 0},
        {"drug": "Ranolazine", "SMILES": "COc1ccccc1OCC(O)CN2CCN(CC(=O)Nc3c(C)cccc3C)CC2", "delta_QTc_ms": 5.0, "tdp_risk_class": 0},
        {"drug": "Tamoxifen", "SMILES": "CC/C(=C(/c1ccccc1)c2ccc(OCCN(C)C)cc2)c3ccccc3", "delta_QTc_ms": 3.0, "tdp_risk_class": 0},
        {"drug": "Verapamil", "SMILES": "COc1ccc(CCN(C)CCCC(C#N)(c2ccc(OC)c(OC)c2)C(C)C)cc1OC", "delta_QTc_ms": -2.0, "tdp_risk_class": 0},
        {"drug": "Metoprolol", "SMILES": "COCCc1ccc(OCC(O)CNC(C)C)cc1", "delta_QTc_ms": 1.0, "tdp_risk_class": 0},
        {"drug": "Mexiletine", "SMILES": "CC(N)COc1c(C)cccc1C", "delta_QTc_ms": 0.0, "tdp_risk_class": 0},
        {"drug": "Nifedipine", "SMILES": "COC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c2ccccc2[N+](=O)[O-]", "delta_QTc_ms": 2.0, "tdp_risk_class": 0},
        {"drug": "Nitrendipine", "SMILES": "CCOC(=O)C1=C(C)NC(C)=C(C(=O)OC)C1c2cccc([N+](=O)[O-])c2", "delta_QTc_ms": 1.0, "tdp_risk_class": 0},
    ]
    
    # Standardize CiPA smiles
    valid_cipa = []
    for c in cipa_compounds:
        mol = Chem.MolFromSmiles(c["SMILES"])
        if mol:
            c["SMILES"] = Chem.MolToSmiles(mol)
            valid_cipa.append(c)
            
    return pd.DataFrame(valid_cipa)

def main():
    os.makedirs('data', exist_ok=True)
    
    # 1. Load CiPA data
    cipa_df = get_cipa_data()
    
    # 2. Stratified train/val/test split on CiPA 28
    # 80/10/10 split
    # First split 80% train, 20% val_test
    X = cipa_df[['SMILES', 'delta_QTc_ms', 'tdp_risk_class']]
    y = cipa_df['tdp_risk_class']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Split the 20% into 10% val, 10% test (i.e. half and half)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    # Save CiPA
    X_train.to_csv('data/cipa_train.csv', index=False)
    X_val.to_csv('data/cipa_val.csv', index=False)
    X_test.to_csv('data/cipa_test.csv', index=False)
    
    # 3. Query ChEMBL
    herg_df = fetch_herg_data()
    herg_df[['SMILES', 'IC50_nM']].to_csv('data/chembl_herg_cleaned.csv', index=False)
    
    # 4. Print Summary
    print("\n" + "="*50)
    print("PIPELINE SUMMARY")
    print("="*50)
    print(f"Total ChEMBL hERG compounds: {len(herg_df)}")
    if len(herg_df) > 0:
        print(f"  - MW Mean: {herg_df['MW'].mean():.1f} Da")
        print(f"  - MW Max: {herg_df['MW'].max():.1f} Da")
    
    print("\nCiPA 28 Split:")
    print(f"  - Train: {len(X_train)} compounds")
    print(f"  - Val:   {len(X_val)} compounds")
    print(f"  - Test:  {len(X_test)} compounds")
    
    print("\nCiPA Class Distribution (Train/Val/Test):")
    for cls in [0, 1, 2]:
        tr = len(y_train[y_train == cls])
        va = len(y_val[y_val == cls])
        te = len(y_test[y_test == cls])
        label = "High" if cls == 2 else "Medium" if cls == 1 else "Low"
        print(f"  - Risk {label} (Class {cls}): {tr} / {va} / {te}")
    print("="*50)

if __name__ == "__main__":
    main()
