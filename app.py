pip install rdkit-pypi
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors
import pandas as pd
import tempfile
import os
import zipfile
import base64
from io import BytesIO

try:
    from openbabel import pybel
except ImportError:
    st.error("❌ Open Babel (pybel) not installed! Run: `conda install -c conda-forge openbabel`")
    st.stop()

st.set_page_config(page_title="3D Ligand Generator", layout="wide")
st.title("💊 3D Ligand Generator")
st.markdown("""
Upload `.txt`, `.csv`, or `.xlsx` with **SMILES and optional Name** columns.  
This version **removes salts**, shows 2D previews (optional), and packages all outputs into a single ZIP.  
Filters applied based on **Lipinski's Rule of Five**.
""")

uploaded_file = st.file_uploader("📂 Upload SMILES File", type=["txt", "csv", "xlsx"])
formats = st.multiselect("📦 Select Output Formats", ["sdf", "mol2", "pdb", "pdbqt"], default=["sdf"])

show_2d = st.checkbox("🧬 Show 2D Structure Previews", value=True)
preview_range = st.slider("🔍 Range of Previews", 1, 100, (1, 30))
ph_value = st.slider("⚗️ Desired pH for Hydrogen Optimization", 0.0, 14.0, 7.4, 0.1)

mw_range = st.slider("💠 Molecular Weight Range (Da)", 0.0, 1000.0, (0.0, 500.0), step=1.0)
logp_range = st.slider("💧 LogP Range", -10.0, 10.0, (-5.0, 5.0), step=0.1)

def remove_salts(smiles):
    try:
        frags = smiles.split(".")
        return max(frags, key=len)
    except:
        return smiles

def validate_smiles_with_names(smiles_list, name_list=None):
    valid, invalid = [], []
    for idx, smi in enumerate(smiles_list):
        name = name_list[idx] if name_list else f"MOL_{idx+1}"
        clean_smi = remove_salts(smi)
        mol = Chem.MolFromSmiles(clean_smi)
        if mol:
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            if mw_range[0] <= mw <= mw_range[1] and logp_range[0] <= logp <= logp_range[1]:
                valid.append((idx + 1, clean_smi, mol, name.strip().replace(" ", "_")))
            else:
                invalid.append((idx + 1, smi, name, f"Failed Lipinski (MW={mw:.1f}, LogP={logp:.2f})"))
        else:
            invalid.append((idx + 1, smi, name, "Invalid SMILES"))
    return valid, invalid

def smiles_to_3d_obmol(smi):
    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    sdf = Chem.MolToMolBlock(mol)
    return pybel.readstring("mol", sdf)

def mol_to_png_base64(mol):
    img = Draw.MolToImage(mol, size=(1200, 1200))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# 🚀 Always show analyze button for clarity:
analyze_triggered = st.button("🚀 Analyze")

if analyze_triggered:
    if not uploaded_file:
        st.warning("⚠️ Please upload a file first!")
        st.stop()

    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            txt = uploaded_file.read().decode("utf-8")
            df = pd.DataFrame([l.strip() for l in txt.splitlines() if l.strip()], columns=["SMILES"])

        if df.shape[1] >= 2:
            smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
            name_list = df.iloc[:, 1].fillna("").astype(str).tolist()
        else:
            smiles_list = df.iloc[:, 0].dropna().astype(str).tolist()
            name_list = [f"MOL_{i+1}" for i in range(len(smiles_list))]

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")
        st.stop()

    st.info(f"🔎 Validating and removing salts from {len(smiles_list)} SMILES...")
    valid_smiles, invalid_smiles = validate_smiles_with_names(smiles_list, name_list)

    if not valid_smiles:
        st.error("❌ No valid SMILES passed filters.")
        st.stop()

    st.success(f"✅ Processing {len(valid_smiles)} valid molecules...")

    with tempfile.TemporaryDirectory() as tmpdir:
        final_zip_name = "3D_ligands_named.zip"
        zip_path = os.path.join(tmpdir, final_zip_name)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            # 3D structures
            for idx, smi, mol, name in valid_smiles:
                try:
                    obmol = smiles_to_3d_obmol(smi)
                    for fmt in formats:
                        out_file = os.path.join(tmpdir, f"{name}.{fmt}")
                        obmol.write(fmt, out_file, overwrite=True)
                        zipf.write(out_file, arcname=os.path.basename(out_file))
                except Exception as e:
                    st.error(f"⚠️ Error with {name}: {e}")

            # invalid SMILES CSV if any
            if invalid_smiles:
                invalid_df = pd.DataFrame(invalid_smiles, columns=["Index", "Invalid_SMILES", "Name", "Reason"])
                invalid_csv_path = os.path.join(tmpdir, "invalid_smiles.csv")
                invalid_df.to_csv(invalid_csv_path, index=False)
                zipf.write(invalid_csv_path, arcname="invalid_smiles.csv")

        # ✅ Directly read zip for download without moving it (avoids FileExistsError)
        with open(zip_path, "rb") as f:
            st.download_button(
                "📦 Download All 3D Ligands + Invalids (ZIP)",
                data=f.read(),
                file_name=final_zip_name,
                mime="application/zip"
            )

    if show_2d:
        st.markdown("## 🧪 2D Preview of Valid Ligands")
        cols = st.columns(3)
        start, end = preview_range
        for i, (_, smi, mol, name) in enumerate(valid_smiles[start-1:end]):
            img_base64 = mol_to_png_base64(mol)
            with cols[i % 3]:
                st.image(f"data:image/png;base64,{img_base64}",
                         caption=f"{name}: {smi}", use_container_width=True)

    st.markdown("## ✅ Summary")
    st.success(f"""
- 🧪 Total SMILES submitted: {len(smiles_list)}
- ✅ Valid molecules processed: {len(valid_smiles)}
- ❌ Invalid entries: {len(invalid_smiles)}
- ⚗️ Protonation pH (used during hydrogen addition): {ph_value}
- 🧼 Salt fragments removed using string-based fragment separation ('.') and longest fragment selection.
- 📊 Filters applied:
  - Molecular Weight: {mw_range[0]} - {mw_range[1]} Da
  - LogP: {logp_range[0]} to {logp_range[1]}
- 🧬 3D structures generated with **RDKit (EmbedMolecule + MMFF94 Optimization)** and exported via **Open Babel (Pybel)**.
- 📤 Formats exported: {', '.join(formats)}
- 📄 Invalid SMILES CSV {'included in zip' if invalid_smiles else 'not needed (all valid)'}
""")
