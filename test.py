import numpy as np
import librosa
import tensorflow as tf
import os

# CONFIGURATION
# le fichier.keras
DOSSIER_ACTUEL = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DOSSIER_ACTUEL, "detecteur_urgence_final.keras")

SR = 22050
DURATION = 4
N_MELS = 128
EXPECTED_WIDTH = 173

# CHARGEMENT DU MODELE
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Succès : Modèle chargé depuis {MODEL_PATH}")
else:
    print(f"Erreur : Le fichier {MODEL_PATH} est introuvable.")

def predire_audio(chemin_fichier):
    try:
        # Chargement du signal audio
        audio, _ = librosa.load(chemin_fichier, sr=SR, res_type='kaiser_fast')
        
        #Supprimer le silence
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        # Ajustement de la duree a 4 secondes exactes
        target_len = SR * DURATION
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))), mode='constant')
        
        # Extraction du Spectrogramme de Mel
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Redimensionnement de la largeur temporelle
        if mel_spec_db.shape[1] < EXPECTED_WIDTH:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, EXPECTED_WIDTH - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :EXPECTED_WIDTH]
            
        # Normalisation Min-Max (0 a 1)
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-6)
        
        # Preparation pour le CNN 
        input_tensor = mel_spec_db[np.newaxis, ..., np.newaxis]
        
        # Prediction
        prediction_score = model.predict(input_tensor, verbose=0)[0][0]
        
        # Affichage du resultat
        print("-" * 50)
        print(f"Analyse du fichier : {os.path.basename(chemin_fichier)}")
        
        if prediction_score > 0.6:
            print("RESULTAT :  ALERTE SON D'URGENCE DETECTE")
            print(f"Indice de confiance : {prediction_score:.2%}")
        else:
            print("RESULTAT : SITUATION NORMALE")
            print(f"Probabilite d'urgence : {prediction_score:.2%}")
        print("-" * 50)
        
        return prediction_score

    except Exception as e:
        print(f"Erreur lors de l'analyse du fichier : {e}")
        return None

# EXEMPLE D'UTILISATION 
mon_fichier_test = os.path.join(DOSSIER_ACTUEL, "83488-1-1-0.wav") 

if os.path.exists(mon_fichier_test):

    predire_audio(mon_fichier_test)
else:
    print(f"Veuillez spécifier un fichier valide. Fichier cherché : {mon_fichier_test}")
