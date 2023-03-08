# Import des bibliothèques nécessaires
from datasets import load_from_disk

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import torch.nn as nn

from transformers.adapters import BertAdapterModel

from transformers import (
    AdapterConfig,
    BertTokenizer,
    TrainingArguments,
    AdapterTrainer,
    EarlyStoppingCallback
)

# On crée le tokenizer à partir de bert-base-multilingual-cased
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Fonction pour encoder les données d'un batch
def encode_batch(batch):
    return tokenizer(batch["text"], max_length=256, truncation=True, padding="max_length")

# Fonction pour calculer les métriques
def compute_metrics(p):
    labels = p.label_ids
    preds = p.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# On utilise bert-base-multilingual-cased comme modèle de base
model = BertAdapterModel.from_pretrained("bert-base-multilingual-cased")
        
# On charge l'architecture de l'adapter qu'on va utiliser
adapter_config = AdapterConfig.load("houlsby", non_linearity="gelu", reduction_factor=96)
        
# On ajoute l'adapter au modèle
model.add_adapter("naelsenAdapter", config=adapter_config)
        
# On gèle tous les paramètres sauf ceux de l'adapter
model.train_adapter("naelsenAdapter")
        
# On utilise l'adapter dans toutes les phases forwards
model.set_active_adapters("naelsenAdapter")
        
# On ajoute la tête pour fine-tuner le modèle
model.add_classification_head(
    "naelsenAdapterHead",
    layers=2,
    num_labels=2,
    id2label={ 0: "👎", 1: "👍"}
)

# On affiche le nombre de paramètres entraînables
trainable_prams = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Nombre de paramètres entraînables : {trainable_prams}')

# On récupère et pré-traite le dataset pour préparer l'entraînement
dataset = load_from_disk("dataset_100")
dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# On définit le callback pour l'early stopping
early_stopping = EarlyStoppingCallback(early_stopping_patience=3)

# On définit les arguments pour l'entraînement
training_args = TrainingArguments(
    learning_rate=1e-4,
    lr_scheduler_type="cosine_with_restarts",
    warmup_steps=100,
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir="./training_output_adapter",
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model='f1',
    greater_is_better=True,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

# On définit le processus d'entraînement du modèle
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    callbacks=[early_stopping]
)

# Entraînement du modèle
train_result = trainer.train()

# On enregistre les configs et les poids de l'adapter et de sa tête de classification
model.save_adapter("./naelsenAdapter", "naelsenAdapter")
model.save_head("./naelsenAdapterHead", "naelsenAdapterHead")
