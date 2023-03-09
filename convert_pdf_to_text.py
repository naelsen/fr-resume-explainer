from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import re

def convert_pdf_to_text(file_path):
    
    # On crée un gestionnaire de ressources PDF
    resource_manager = PDFResourceManager()
    
    # On crée un tampon de texte en sortie
    output_string = StringIO()
    
    # On crée un convertisseur de PDF en texte
    converter = TextConverter(resource_manager, output_string, laparams=LAParams())

    # On crée un interpréteur de pages PDF
    interpreter = PDFPageInterpreter(resource_manager, converter)
    
    # On ouvre le fichier PDF en mode binaire
    with open(file_path, 'rb') as file:
        # On traite chaque page du PDF
        for page in PDFPage.get_pages(file):
            interpreter.process_page(page)
            
    # On récupère le texte converti
    text = output_string.getvalue()
    text = ' '.join(re.findall("\w[\w\.]*@\w+\.\w+|(?:(?:https?|ftp):\/\/)?[\w/\-?=%.]+\.[\w/\-&?=%.]+|[\w']+", text))

    # On ferme le tampon de sortie
    output_string.close()
    
    return text
