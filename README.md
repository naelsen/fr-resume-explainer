# fr-resume-explainer

`fr-cv-explainer` est un dépôt Github permettant de classer les CV français et expliquer pourquoi il est adapté ou pas pour postuler à un poste de Data Scientist. Pour cela, j'ai construit une base de données en utilisant des techniques de Web Scraping. J'ai scrapé de manière responsable les offres d'emploi Indeed, j'ai valorisé ces données. En outre, j'ai utilisé l'architecture BERT (bert-base-multilingual-cased) et les Adapters de la librairie `adapter-transformers` pour construire et entraîner mon modèle. Les prédictions du modèle sont expliquées grâce à l'estimation des valeurs de Shapley.

## Installation

`fr-resume-explainer` supporte **Python 3.8+**.

_Important 💡 : Assurez-vous de créer un nouvel environnement Python 3.8+ avant de commencer._

```
git clone https://github.com/naelsen/fr-resume-explainer.git
cd fr-resume-explainer
pip install -r requirements.txt
```

## Comment analyser votre CV ?

```
python3 predict_and_explain_cv.py <YOUR_CV_FILE.pdf>
```

## Comment construire votre Dataset ?

### 1) [Générer une clef API OpenAI](https://platform.openai.com/account/api-keys).

### 2) Par sécurité, stocker cette clef dans une variable d'environnement :

- Ouvrez le fichier caché bashrc avec votre éditeur de texte préféré, par exemple :

```
gedit ~/.bashrc
```

- Ajouter cette ligne dans le fichier

```
export OPENAI_API_KEY=<YOUR_KEY>
```

- Vous pouvez maintenant modifier le fichier `build_dataset.py` pour mettre plusieurs clés si vous disposez de plusieurs comptes OpenAPI. Il est important que chaque compte dispose d'un numéro de téléphone unique pour disposer des 18 $ de jeton gratuit. Ensuite, tapez la commande suivante dans votre terminal :
```
python3 build_dataset.py
```

## Comment construire le modèle ?

Vous pouvez lancer la commande suivante :
```
python3 build_adapter-bert-based.py 
```

## Architecture du modèle

J'ai utilisé l'implémentation de BERT dans les librairies transformer et adapter-transformers pour construire le modèle. J'ai fine-tuné le modèle BERT-base-multilingual-cased en ajoutant un adapter et une tête de classification au modèle BERT. Les Adapters sont l'état de l'art en Transfer Learning, ce sont des réseaux neuronaux qui peuvent être ajoutés à une architecture de modèle pré-entraîné, dans notre cas à BERT. Ils introduisent une petite quantité de nouveaux paramètres Φ dans le modèle pré-entraîné avec des paramètres Θ existants. Les paramètres Φ sont entraînables pour une tâche spécifique tout en conservant Θ fixe, de sorte qu'ils apprennent à encoder des représentations spécifiques à la tâche dans les couches intermédiaires du modèle pré-entraîné.

## Évaluation du modèle

Le modèle a été entaîné pour minimiser la fonction d'entropie croisée de l'ensemble d'entraînement et a été évalué sur l'ensemble maximisant la métrique F1 sur un ensemble de validation.

![evaluation](https://github.com/naelsen/fr-resume-explainer/blob/main/tensorBoard.png)

## Interprétation des résultats

Pour interpréter les prédictions du modèle, je vais utiliser la méthode d'éstimation des valeurs de shapley. Cette méthode se base sur la théorie des jeux pour estimer l'importance de chaque feature pour chaque prédiction du modèle. Les valeurs de shapley permettent ainsi d'expliquer les décisions prises par le modèle et d'identifier les features les plus importantes pour la tâche de classification.

Voici un exemple sur mon CV :

![Exemple](https://raw.githubusercontent.com/naelsen/fr-resume-explainer/main/CV_Nael_Sennoun_vf_analyses.png)
