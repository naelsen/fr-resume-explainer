from ScraperDatasetCollector import IndeedScraperDatasetCollector

if __name__ == "__main__":

	# metier utilisé pour ajouter de la "positivité" aux mots clefs de Data Science
	my_job = "data scientist"

	# metier utilisé pour contrealancer la "positivité" des mots clefs
	various_jobs = ["commercial",
	                "medecin generaliste",
	                "infirmier",
	                "chargé de recrutement",
	                "avocat droit social",
	                "architecte d'interieur",
	                "neuropsychologue",
	                "ingenieur en genie civil",
	                "chef de projet informatique",
	                "journaliste",
	                "ingenieur agroalimentaire",
	                "developpeur backend",
	                "citrix"]

	# on utilise des clefs d'API OpenAI stockées dans les variables d'environnements
	# (pour ne pas se faire voler ses clefs API sur Github)
	scrapper = IndeedScraperDatasetCollector(["OPENAI_API_KEY_SOU",
	                                          "OPENAI_API_KEY_KAM",
	                                          "OPENAI_API_KEY_GHA",
	                                          "OPENAI_API_KEY_NOA"])

	# nombre de pages à scrapper
	n_pages = 100

	scrapper.main(my_job, various_jobs, n_pages)
