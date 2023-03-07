# pour accéder à la variable d'environnement dans le ~/.bashrc 
import os
# supprimer un dossier et son contenu
from shutil import rmtree
# pour randomiser les choix d'URL
import random
# pour traiter le texte
import re
# pour garantir des delais afin de paraitre plus humain
import time

# pour créer des dataset plus facilement
from datasets import Dataset
# pour récupérer la barre de progrès d'une boucle
from tqdm import tqdm
# pour scrapper
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, StaleElementReferenceException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
# pour accéder à l'API OpenAI
import openai

print("""
#########################################################
##  __                  __          __                 ##
## (_  _ _ _  _  _  _ _|  \ _ |_ _ /   _ || _ _|_ _  _ ##
## __)(_| (_||_)|_)(-| |__/(_||_(_|\__(_)||(-(_|_(_)|  ##
##           |  |                                      ##
##     ----------------->  https://github.com/naelsen  ##
##                                                     ##
#########################################################
""")

class IndeedScraperDatasetCollector:

    # Initialisation des variables de classe
    def __init__(self, OPENAI_API_KEYs):

        # récupérer une clé API d'OpenAI à partir de variable d'environnement
        self.OPENAI_API_KEYs = OPENAI_API_KEYs
        api_key = os.getenv(self.OPENAI_API_KEYs[0])
        openai.api_key = api_key

        # User agent pour la requête HTTP pour tromper le site web que le scraper est un navigateur Web
        self.user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
                        (KHTML, like Gecko) Chrome/110.0.5481.77 Safari/537.36'

        # options pour le navigateur
        self.options = webdriver.ChromeOptions()
        self.options.add_argument('user-agent={' + self.user_agent + '}')

        # initialisation du navigateur Chrome
        self.driver = webdriver.Chrome(executable_path = './chromedriver_linux64/chromedriver',
                                       options = self.options)

        # temps d'attente maximum en seconde fixé par le driver
        self.wait = WebDriverWait(self.driver, 10)

        # initialisation de la liste de fiches de postes et de la liste de logs
        self.records = []
        self.log = [None, None]

        # Accéder à la page d'accueil d'Indeed et supprimez tous les cookies
        self.driver.get("https://fr.indeed.com/")
        self.driver.delete_all_cookies()

    # retourne une liste d'URLs correspondant aux différentes offres d'emploi
    def __init_jobs_url(self, my_job, other_jobs):

        # définission de l'expression régulière pour remplacer les espaces dans l'URL
        match = re.compile(' ')

        # définission du template d'URL pour accéder aux offres d'emploi
        url_template = "https://fr.indeed.com/jobs?q={}&l=france"

        # création d'un dictionnaire contenant le prochain URL à accéder pour mon emploi
        my_job_url = {"my_job" : url_template.format(match.sub('+',my_job))}

        # création d'un dictionnaire contenant la liste des prochains URLs à accéder des autres emplois
        other_jobs_urls = {"other_jobs" : {position : url_template.format(match.sub('+',position)) \
                                           for position in other_jobs}}

        # fusion des dictionnaires
        jobs_urls = {**my_job_url, **other_jobs_urls}

        return jobs_urls

    # récupère le prochain URL que le driver va accéder de manière aléatoire 
    def __get_next_url(self, jobs_urls):

        # n est choisi au hasard : soit 0 soit 1 
        n = random.randint(0,1)
        # si n est égale à 0 : le prochain URL récupérer concernera mon emploi
        if n == 0:

            # on enregistre dans le log que la prochaine page à scrapper concerne mon emploi
            self.log[0] = "my_job"
            self.log[1] = None
            next_url = jobs_urls[self.log[0]]

        # si n est égale à 1 : le prochain URL récupérer concernera un emploi qui n'est pas le mien
        else:
            valid_other_jobs = [key for key in jobs_urls["other_jobs"] if jobs_urls["other_jobs"][key] is not None]

            # indique qu'on a atteint la fin de toutes les pages associées aux autres jobs
            if len(valid_other_jobs) == 0:
                next_url = None
            # on récupère aléatoirement le prochain url d'un des autres job
            else:
                # on enregistre dans le log qu'on va scrapper un autre emploi choisi au hasard dans la liste
                self.log[0] = "other_jobs"
                self.log[1] = random.choice(valid_other_jobs)
                next_url = jobs_urls[self.log[0]][self.log[1]]

        return next_url

    # récupère les compétence techniques selon la description de l'offre d'emploi
    def __get_skills_from_description(self, description):

        # prompt engineering pour récupérer les skills à partir d'une description
        prompt_engineered = "Tu es un expert en recrutement. " \
        "Tu aides les entreprises à recruter les meilleurs profils depuis 20 ans. " \
        "Ton rôle consiste à conserver UNIQUEMENT les hards SKILLS nécessaire pour les offres d'emploi des entreprises. " \
        "Commence avec cette fiche de poste :\n{}\n" \
        "Hard skills keywords :\n"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt_engineered.format(description),
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        txt = response.choices[0].text

        # on récupère le texte nettoyé
        txt = ' '.join(re.findall("[\w']+", txt)).lower()

        return txt


    def __add_data_from_description(self, description):

        skills = self.__get_skills_from_description(description)

        # si la description concerne mon emploi on créer deux groupes de données labélisé 1 et 0
        if self.log[0]=="my_job":

            # prompt engineering pour créer 3 phrases expliquant ne pas avoir les compétences recquises
            prompt_engineered_neg = "Compétences techniques :\n{}" \
            "Voici ci-dessous 3 phrases (1. 2. et 3.) avec une entropie maximisée par les compétences techniques ci-dessus, " \
            "ces 3 phrases expliquent pourquoi je NE possède PAS les compétences techniques ci-dessus :\n"

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt_engineered_neg.format(skills),
                temperature=1,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            txts = response.choices[0].text

            # créer une liste des 3 phrases labelisées 0
            txts_neg = re.findall("[1-3]\..+", txts)

            # prompt engineering pour créer 3 phrases expliquant ne pas avoir les compétences recquises
            prompt_engineered_pos = "Compétences techniques :\n{}" \
            "Voici ci-dessous 3 phrases (1. 2. et 3.) avec une entropie maximisée par les compétences techniques ci-dessus, " \
            "ces 3 phrases expliquent pourquoi je possède les compétences techniques ci-dessus :\n"

            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt_engineered_pos.format(skills),
                temperature=1,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            txts = response.choices[0].text

            # créer une liste des 3 phrases labelisées 1
            txts_pos = re.findall("[1-3]\..+", txts)

            # on ajoute toutes les données à la fin au cas ou l'on dépasse la capacité de calcul offerte par OPENAI 
            for txt in txts_pos:
                txt = ' '.join(re.findall("(?<=\s)[\w']+", txt)).lower()
                self.records.append({"text":txt, "label":1})

            for txt in txts_neg:
                txt = ' '.join(re.findall("(?<=\s)[\w']+", txt)).lower()
                self.records.append({"text":txt, "label":0})

        # ajoute les mots clefs labelisé 1 s'ils concernent mon emploi et 0 sinon
        self.records.append({"text":skills, "label":(self.log[0]=="my_job")*1})

    # ajouter la fiche de poste labélisé à la liste des fiches
    def __add_one_record(self, job):

        # attendre un temps aléatoire avant fermer les fenêtres pop-up s'il y en a
        if self.log[0]=="my_job":
            time.sleep(random.uniform(0.2, 1.8))
        else:
            time.sleep(random.uniform(1.8, 6.2))
        try:
            close_buttons = self.driver.find_elements(By.XPATH, '//button[contains(@class,"icl-CloseButton")]')
            for button in close_buttons:
                button.click()
        except:
            pass

        # scroller jusqu'au prochain emploi selectionné sur la page
        job.location_once_scrolled_into_view
        if job.get_attribute("href") is None:
            job.click()
        else:
            print("pass")
            return
        # pas de description pour l'instant
        job_description = None

        try:
            # attendre que la description de l'emploi soit visible
            self.wait.until(EC.visibility_of_element_located(
                (By.XPATH, '//div[@id = "jobDescriptionText"]')
            ))

            # extraire la description de l'emploi, nettoyer et traiter le texte
            job_description = self.driver.find_element(By.XPATH, '//div[@id = "jobDescriptionText"]').text.strip()

        # si la description de l'emploi n'est pas visible on passe
        except TimeoutException:
            pass

        # si on a récupérer la description de l'offre d'emploi on entre dans la condition
        if job_description is not None:

            # on remplace les sauts de lignes, espaces, tabulations... par un seul espace
            job_description = re.sub('\s+', ' ', job_description)

            # ajoute les données liées à la description de l'offre d'emploi
            try:
                self.__add_data_from_description(job_description)

            # si on ne peut plus utiliser la capacité de calcul d'OPENAI on change de compte grace aux clefs api 
            except:
                print("{} : RateLimitError".format(self.OPENAI_API_KEYs[0]))
                self.OPENAI_API_KEYs = self.OPENAI_API_KEYs[1:]
                if len(self.OPENAI_API_KEYs) != 0:
                    api_key = os.getenv(self.OPENAI_API_KEYs[0])
                    openai.api_key = api_key
                    print("Switch to {}".format(self.OPENAI_API_KEYs[0]))

                    # on réessaye la fonction avec la nouvelle clef
                    self.__add_one_record(job)

    # ajouter toutes les fiches de postes de la page associé à l'URL
    def __add_records_from_url(self, url, retried = False):

        # naviguer vers l'URL spécifiée et gérer les problèmes de connections
        try:
            self.driver.get(url)
        except:
            print("Probème de connection... Attente de 30 secondes avant reconnection...")
            time.sleep(30)
            print("Reconnection...")
            self.__add_records_from_url(url, retried)
            return

        # rejeter les cookiers si le pop-up apparaît
        try:
            self.wait.until(
                EC.presence_of_element_located((By.ID, "onetrust-reject-all-handler"))
            ).click()
        except:
            pass

        try:

            # attendre que la liste des emplois soit visible avant d'essayer des les garder en mémoire
            self.wait.until(EC.visibility_of_element_located(
                (By.XPATH,'//div[contains(@class,"job_seen_beacon")]')
            ))
            jobs = self.driver.find_elements(By.XPATH,'//div[contains(@class,"job_seen_beacon")]')

            # ajouter les données concernant les offres d'emplois de la page une par une
            for job in jobs:
                self.__add_one_record(job)

                # si on ne peut plus utiliser de clefs API on arrête
                if len(self.OPENAI_API_KEYs) == 0:
                    break

        # réessayer une fois si la liste des emplois ne se rend pas visible
        except TimeoutException:
            if not retried:
                print("TimeoutException error : retrying to get info on current page...")
                self.__add_records_from_url(url, True)

        except StaleElementReferenceException:
            pass

    # met à jour le dictionnaire des URLs en modifiant le prochain URL à visiter selon le log
    def __update_jobs_url(self, jobs_urls):

        # est ce qu'on peut aller à la page suivante actuelle ?
        try:
            new_url = self.wait.until(EC.visibility_of_element_located(
                (By.XPATH, './/a[contains(@aria-label,"Next")]')
            )).get_attribute('href')

        # si on ne peut peut pas récupérer l'URL de la page suivante actuelle
        except TimeoutException:
            new_url = None

        # on précise le nouveau URL à traiter la prochaine fois selon le log
        if self.log[1] is None:
            jobs_urls[self.log[0]] = new_url
        else:
            jobs_urls[self.log[0]][self.log[1]] = new_url


    def __save_dataset_to_disk(self, dataset_name):

        # création d'un dataset à partir de la liste "records" récupérée par le web scrapper
        dataset = Dataset.from_list(self.records)

        # séparation du dataset en deux parties :
        # - une partie pour l'entraînement du modèle
        # - une partie pour les tests du modèle
        dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

        # enregistrement du dataset dans un fichier sur le disque dur
        dataset.save_to_disk(dataset_name)

    def __save_dataset_checkpoint(self, page):
        if os.path.exists("dataset_{}".format(page-1)):
            rmtree("dataset_{}".format(page-1))
        self.__save_dataset_to_disk("dataset_{}".format(page))


    def main(self, my_job, other_jobs, n_pages = 10):

        # initialise les URLs des offres d'emploi
        jobs_urls = self.__init_jobs_url(my_job, other_jobs)

        # récupère l'URL de la prochaine page d'offres d'emploi à récupérer
        for page in tqdm(range(1, n_pages+1)):
            url = self.__get_next_url(jobs_urls)

            # bérifie si toutes les pages ont été récupérées
            if url is None:
                print("There is no more data to scrap.")
                self.__save_dataset_checkpoint(page)
                break

            # récupère les données de la page spécifiée et les stockes
            self.__add_records_from_url(url)

            # si on ne peut plus utiliser de clefs API on arrête
            if len(self.OPENAI_API_KEYs) == 0:
                print("No more OpenAI API keys to use.")
                self.__save_dataset_checkpoint(page)
                break

            # met à jour la liste des URLs des offres d'emploi
            self.__update_jobs_url(jobs_urls)

            # pour ne pas perdre toutes les données en cas d'erreur
            self.__save_dataset_checkpoint(page)
            
        # ferme le navigateur
        self.driver.quit()
