# fr-resume-explainer

Ce projet vise à classer les CV en fonction de leur compétence en Data Science. Pour cela, j'ai utilisé l'architecture BERT et les Adapters de la librairie `adapter-transformers` afin d'améliorer les performances du modèle en termes de transfert de connaissances.

## Architecture du modèle

J'ai utilisé l'implémentation de BERT dans les librairies transformer et adapter-transformers pour construire le modèle. J'ai fine-tuné le modèle BERT-base-multilingual-cased en ajoutant un adapter et une tête de classification au  modèle BERT. Les Adapters sont des réseaux neuronaux qui peuvent être ajoutés à une architecture de modèle pré-entraîné. Ils introduisent une petite quantité de nouveaux paramètres Φ dans le modèle pré-entraîné avec des paramètres Θ existants. Les paramètres Φ sont entraînable pour une tâche spécifique tout en conservant Θ fixe, de sorte qu'ils apprennent à encoder des représentations spécifiques à la tâche dans les couches intermédiaires du modèle pré-entraîné.

## Évaluation du modèle

Le modèle a été entaîné pour minimiser la fonction d'entropie croisée de l'ensemble d'entraînement et a été évalué sur l'ensemble maximisant la métrique F1 sur un ensemble de validation.

![evaluation](https://github.com/naelsen/fr-resume-explainer/blob/main/tensorBoard.png)

## Interprétation des résultats

Pour interpréter les prédictions du modèle, j'ai utilisé la méthode des valeurs de shapley. Cette méthode se base sur la théorie des jeux pour estimer l'importance de chaque feature pour chaque prédiction du modèle. Les valeurs de shapley permettent ainsi d'expliquer les décisions prises par le modèle et d'identifier les features les plus importants pour la tâche de classification.

## Installation

`fr-resume-explainer` supporte **Python 3.8+**.

```
git clone https://github.com/naelsen/fr-resume-explainer.git
cd fr-resume-explainer
pip install -r requirements.txt
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

- Vous pouvez maintenant modifer et lancer `build_dataset.py` avec plusieurs clefs si vous disposez de plusieurs compte OpenAPI. Il est important que chaque compte dispose d'un numéro de téléphone unique pour disposer des 18$ de jeton gratuit.

## Comment construire le mod ?

