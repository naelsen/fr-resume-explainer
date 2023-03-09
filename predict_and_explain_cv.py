# Import des bibliothèques nécessaires
from convert_pdf_to_text import convert_pdf_to_text
from transformers import BertTokenizer, TextClassificationPipeline
from transformers.adapters import BertAdapterModel
import torch
import re
import argparse
import matplotlib.pyplot as plt
import shap

# Définition d'une fonction pour faire des prédictions sur l'ensemble des segments de texte du cv
def make_predictions(classifier, X_batch_text):

    # Utilisation du modèle pour faire des prédictions sur les segments de texte
    preds = classifier(X_batch_text)

    # Extraction des étiquettes prédites et des scores associés
    labels = [p['label'] for p in preds]
    scores = [p['score'] for p in preds]

    # Calcul du nombre de prédictions positives et négatives
    count_positive = labels.count('👍')
    count_negative = labels.count('👎')

    # Calcul du score total pour les prédictions positives et négatives
    score_positive = sum([s for s, l in zip(scores, labels) if l == '👍'])
    score_negative = sum([s for s, l in zip(scores, labels) if l == '👎'])

    # Retourne "👍" si le nombre de prédictions positives est supérieur au nombre de prédictions négatives,
    # ou si les nombres de prédictions sont égaux mais le score total pour les prédictions positives est supérieur
    if count_positive > count_negative or (count_positive == count_negative and score_positive > score_negative):
        return "👍"
    else:
        return "👎"


# Parsing des arguments en ligne de commande
parser = argparse.ArgumentParser()
parser.add_argument("cv_name", help="Ton cv en pdf")
args = parser.parse_args()

# Chargement de bert-base-multilingual-cased et de mon adapter et ma tête de classification fine tuné
model = BertAdapterModel.from_pretrained("bert-base-multilingual-cased")
model.load_adapter("naelsenAdapter")
model.set_active_adapters("naelsenAdapter")
model.load_head("naelsenAdapterHead")

# Création du tokenizer à partir de bert-base-multilingual-cased
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Utilisation de la carte graphique si elle est disponible, sinon utilisation du CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Création de la pipeline de classification de texte
classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

# Définition de la longueur maximale d'un segment de texte (en nombre de tokens)
max_length = 256
# Définition du chevauchement entre deux segments de texte (en nombre de tokens)
stride = 0
# Calcul de la longueur maximale d'un segment de texte moins la taille du chevauchement
max_length_minus_stride = (max_length - stride)

# Conversion du CV en format texte
text = convert_pdf_to_text(args.cv_name)

# Tokenisation du texte en segments de longueur maximale "max_length" avec un chevauchement "stride"
tokens = tokenizer.tokenize(text)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
segments = []
while len(input_ids) > 0:
    segments.append(tokenizer.decode(input_ids[:max_length]).lower())
    input_ids = input_ids[max_length_minus_stride:]

# Traitement des segments, on maximise la taille du dernier segment, puis celle du permier afin d'avoir
# le plus d'information pertinente dans ces segments (car il y a plus d'info pertinente en milieu de CV)    
if len(segments) == 2:
    segments = [segments[0] + segments[1]]
elif len(segments) == 3:
    segments = [segments[0], segments[1] + segments[2]]
elif len(segments) == 4:
    segments = [segments[0] + segments[1], segments[2] + segments[3]]
else:
    segments = [segments[0] + segments[1]] + segments[2:-2] + [segments[-2] + segments[-1]]

# Classification du CV
print("\nClassification :\n")
print(args.cv_name + " : {}".format(make_predictions(classifier, segments)))

# Explication de la classification
explainer = shap.Explainer(classifier)
for i, segment in enumerate(segments):
    shap_values = explainer([segment])
    fig = shap.plots.text(shap_values, num_starting_labels=0, display=False)
    with open(f'{args.cv_name[:-4]}_segment_{i}.html','w') as f:
        f.write(fig)
