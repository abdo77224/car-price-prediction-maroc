import streamlit as st
import pandas as pd
import numpy as np
import joblib

# === Charger le dataset original pour l'interface utilisateur ===
df = pd.read_csv("AvitoVoitures_net.csv")  # Version non encodée pour affichage

# === Charger le modèle, l'encoder et le scaler ===
model = joblib.load("gradient_boosting_model.pkl")
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

# === Interface utilisateur ===
st.title("🚗 Prédiction du Prix d'une Voiture")
st.write("Choisissez les caractéristiques de votre voiture pour estimer son prix.")

# Sélectionner les champs
input_data = {}
for col in df.columns:
    if df[col].dtype == "object":  # Variables catégoriques
        input_data[col] = st.selectbox(f"Choisissez une valeur pour {col}", df[col].dropna().unique())
    else:  # Variables numériques
        input_data[col] = st.number_input(f"Entrez une valeur pour {col}", min_value=int(df[col].min()), max_value=int(df[col].max()))

# === Vérification que les noms de colonnes correspondent ===
df_cat = df.select_dtypes(include=['object'])
df_num = df.select_dtypes(include=['int64'])

# Créer un DataFrame utilisateur avec les mêmes colonnes
input_df = pd.DataFrame([input_data])

input_cat = input_df[df_cat.columns]  # Sélection des colonnes catégoriques
input_num = input_df[df_num.columns]  # Sélection des colonnes numériques

# === Appliquer les transformations ===
encoded_input = encoder.transform(input_cat).toarray()
normalized_input = scaler.transform(input_num)

# Fusionner les données transformées
final_input = np.concatenate((normalized_input, encoded_input), axis=1)

# === Prédiction ===
if st.button("Prédire le prix 💰"):
    prediction = model.predict([final_input[0]])
    st.success(f"Prix estimé : {round(prediction[0], 2)} Dh")
