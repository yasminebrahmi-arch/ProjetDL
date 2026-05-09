import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm 

# 1 CONFIGURATION ET PARAMÈTRES

#récupérer le chemin du fichier actuel
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) 
BASE_PATH = CURRENT_DIR
METADATA_PATH = os.path.join(CURRENT_DIR, "UrbanSound8K.csv") 

SR = 22050 # nombre de points audio par seconde(sample rate)        
DURATION = 4         
N_MELS = 128       
EXPECTED_WIDTH = 173 
URGENCE_IDS = [1, 6, 8] #  1 (Klaxon), 6 (Coup de feu), 8 (Sirène) 

#EXTRACTION DE CARACTERISTIQUES
def extract_features(file_name):
    try:
        # Chargement due l'audio
        audio, _ = librosa.load(file_name, sr=SR, res_type='kaiser_fast') 
        
        # Suppression du silence avec TRIM
        audio, _ = librosa.effects.trim(audio, top_db=25)
        
        # Ajustement à 4 secondes exactes
        target_len = SR * DURATION
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, max(0, target_len - len(audio))), mode='constant')

        # Transformation en Spectrogramme de Mel
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Ajustement de la largeur temporelle
        if mel_spec_db.shape[1] < EXPECTED_WIDTH:
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, EXPECTED_WIDTH - mel_spec_db.shape[1])), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :EXPECTED_WIDTH]
            
        # Normalisation Min-Max
        mel_spec_db = (mel_spec_db - np.min(mel_spec_db)) / (np.max(mel_spec_db) - np.min(mel_spec_db) + 1e-6)
        return mel_spec_db.astype(np.float32)
    except Exception as e:

        print(f"Erreur sur le fichier {file_name} : {e}")
        return None

# 2 TRAITEMENT DE DONNÉES

# On vérifie si les sauvegardes existent dans le dossier courant
X_path = os.path.join(CURRENT_DIR, 'X_binary_data.npy') #utilisé pour optimiser la RAM en évitant de recalculer les spectrogrammes à chaque exécution
y_path = os.path.join(CURRENT_DIR, 'y_binary_data.npy')

if os.path.exists(X_path) and os.path.exists(y_path):
    print(" Chargement des données existantes :")
    X = np.load(X_path).astype(np.float32)
    y = np.load(y_path)

else:#le cas ou les fichiers n'existent pas (première execution du code)
    print(f" Recherche du fichier CSV à : {METADATA_PATH}")
    metadata = pd.read_csv(METADATA_PATH)
    features, labels = [], []
    print(" Extraction des caractéristiques.")
    for index, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
        file_path = os.path.join(BASE_PATH, 'fold' + str(row["fold"]), str(row["slice_file_name"]))
        data = extract_features(file_path) #appeler la fonction 
        if data is not None:
            features.append(data)
            labels.append(1 if row["classID"] in URGENCE_IDS else 0)

    X = np.array(features)[..., np.newaxis]
    y = np.array(labels)
    np.save(X_path, X)
    np.save(y_path, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3 ARCHITECTURE CNN
def build_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(), 
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(), 
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

model = build_model((N_MELS, EXPECTED_WIDTH, 1))#création du modèle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
              loss='binary_crossentropy', 
              metrics=['accuracy', tf.keras.metrics.Recall()])

# 4 ENTRAÎNEMENT
class_weights = {0: 1.0, 1: 10.0} # Poids fort pour la classe 1 pour compenser le désiquilibre

#arreter l'entrainement ou réduire le taux d'apprentissage
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

print("\nDébut de l'entraînement...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, 
                    validation_data=(X_test, y_test), 
                    class_weight=class_weights,
                    callbacks=callbacks_list)


# 5 ÉVALUATION ET AFFICHAGE DES MÉTRIQUES
y_pred_prob = model.predict(X_test) 
y_pred = (y_pred_prob > 0.6).astype(int)#seuil = 0.6 

# Rapport de Classification 
print("\n" + "="*45)
print("      RAPPORT DE CLASSIFICATION FINAL")
print("="*45)
print(classification_report(y_test, y_pred, labels=[0, 1], target_names=['NORMAL', 'URGENCE'], zero_division=0))

# Matrice de Confusion 
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NORMAL', 'URGENCE'], 
            yticklabels=['NORMAL', 'URGENCE'])
plt.title('Matrice de Confusion')
plt.ylabel('Vrais Labels')
plt.xlabel('Prédictions')
plt.show()

# Courbes d'Apprentissage 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Graphique de la Perte 
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title('Évolution de la Perte (Loss)')
ax1.set_xlabel('Époques')
ax1.legend()

# Graphique de la Précision 
ax2.plot(history.history['accuracy'], label='Train Acc')
ax2.plot(history.history['val_accuracy'], label='Val Acc')
ax2.set_title('Évolution de la Précision (Accuracy)')
ax2.set_xlabel('Époques')
ax2.legend()
  
plt.tight_layout()
plt.show()

# Sauvegarde
model_save_path = os.path.join(CURRENT_DIR, "detecteur_urgence_final.keras")
model.save(model_save_path)
print(f"\n Modèle sauvegardé avec succès dans : {model_save_path}") 