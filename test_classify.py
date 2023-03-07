# Import des bibliothèques nécessaires
from transformers import BertTokenizer, TextClassificationPipeline
from transformers.adapters import BertAdapterModel
import torch

model = BertAdapterModel.from_pretrained("bert-base-multilingual-cased")
model.load_adapter("naelsenAdapter")
model.set_active_adapters("naelsenAdapter")
model.load_head("naelsenAdapterHead")

# On crée le tokenizer à partir de bert-base-multilingual-cased
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=device)

txts = ["J'ai créer un model mathématiques permettant de détecter si un CV de Data Scientist bon ou pas.",
        "Je suis fort en mathématiques mais je n'ai jamais utilisé de framework de Deep Learning.",
        "Je suis dynamique et sociable, mon point fort c'est la musique."
        ]

print("\nClassification :\n")

for txt in txts:
    class_txt = classifier(txt)[0]
    print("- " + txt + " : " + "{} ({}% of confiance)".format(class_txt["label"], round(class_txt["score"]*100)))