"""Génère compte_rendu.pdf à partir des résultats réels de test_api.py."""
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

BASE_DIR = Path(__file__).resolve().parent
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

QUESTIONS = [
("1. Pourquoi le modèle reçoit-il une entrée de forme (1, 28, 28) ?",
 "Le 1 représente le nombre d’images du batch. Les dimensions 28 × 28 représentent la hauteur et la largeur de l’image Fashion-MNIST."),
("2. Pourquoi l’API vérifie-t-elle la présence de 784 pixels ?",
 "Une image Fashion-MNIST contient 28 × 28 = 784 pixels. Cette vérification garantit que l’image peut être reconstruite correctement."),
("3. Pourquoi faut-il diviser les pixels par 255 lorsque leur valeur maximale dépasse 1 ?",
 "Les pixels d’origine sont codés entre 0 et 255. La division par 255 les ramène entre 0 et 1, comme lors du prétraitement utilisé pour entraîner le modèle."),
("4. Quelle différence existe entre les logits et les probabilités ?",
 "Les logits sont les scores bruts produits par la dernière couche du réseau. Les probabilités sont des valeurs normalisées entre 0 et 1 dont la somme vaut 1."),
("5. Pourquoi applique-t-on une fonction softmax ?",
 "Softmax transforme les logits en probabilités interprétables pour les dix classes. La classe ayant la probabilité la plus élevée est retenue."),
("6. Pourquoi le modèle est-il chargé une seule fois au démarrage du serveur ?",
 "Pour éviter de recharger le fichier à chaque requête, réduire le temps de traitement et réutiliser le modèle en mémoire."),
("7. Quelle différence existe entre model.fit() et model(image, training=False) ?",
 "model.fit() entraîne le réseau en ajustant ses paramètres. model(image, training=False) utilise le modèle déjà entraîné pour effectuer une inférence sans apprentissage."),
("8. Que représente la latence retournée par l’API ?",
 "Elle représente le temps mesuré pour l’appel d’inférence du modèle, en millisecondes. Elle peut varier selon la machine et la charge."),
("9. Quel est le rôle de la route /health ?",
 "Elle permet de vérifier que l’API répond et que le modèle est chargé, ainsi que d’afficher ses formes d’entrée et de sortie."),
("10. Quels éléments constituent le contrat d’inférence de cette application ?",
 "Une requête POST /predict contenant pixels (exactement 784 nombres finis entre 0 et 255). L’API reconstruit et normalise l’image puis renvoie class_id, class_name, confidence et latency_ms. Une entrée invalide est refusée avec une erreur explicite."),
]

def generate_report(rows, summary, bad_result):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="MainTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=7, leading=9))
    story = [
        Paragraph("Compte rendu – TP Déploiement d’un modèle avec FastAPI", styles["MainTitle"]),
        Spacer(1, 0.3*cm),
        Paragraph("<b>Groupe / étudiants :</b> À compléter", styles["BodyText"]),
        Paragraph("<b>Modèle :</b> Fashion-MNIST – réseau MLP fourni (fashion_mnist.keras)", styles["BodyText"]),
        Spacer(1, 0.2*cm),
        Paragraph("1. Objectif et contrat d’inférence", styles["Heading2"]),
        Paragraph("L’application expose un service HTTP permettant de classer une image Fashion-MNIST de 28 × 28 pixels. "
                  "La route POST /predict reçoit un objet JSON contenant une liste pixels de 784 valeurs. "
                  "Elle renvoie l’identifiant et le nom de la classe, la confiance et la latence d’inférence. "
                  "Les dix classes sont : " + ", ".join(CLASS_NAMES) + ".", styles["BodyText"]),
        Spacer(1, 0.2*cm),
        Paragraph("2. Résultats des 20 prédictions", styles["Heading2"]),
    ]
    data = [["Image", "Classe réelle", "Classe prédite", "Confiance", "Latence (ms)", "Résultat"]]
    for r in rows:
        data.append([str(r["image"]), r["classe_reelle"], r["classe_predite"],
                     f'{r["confiance"]:.4f}', f'{r["latence_ms"]:.3f}', r["resultat"]])
    table = Table(data, repeatRows=1, colWidths=[1.0*cm, 3.0*cm, 3.0*cm, 1.8*cm, 2.0*cm, 1.8*cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#263B59")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EEF2F6")]),
    ]))
    story += [table, Spacer(1, 0.2*cm),
        Paragraph(f'<b>Prédictions correctes :</b> {summary["predictions_correctes"]}/{summary["nombre_images"]}', styles["BodyText"]),
        Paragraph(f'<b>Taux de réussite :</b> {summary["taux_reussite_pct"]:.2f} %', styles["BodyText"]),
        Paragraph(f'<b>Latence moyenne :</b> {summary["latence_moyenne_ms"]:.3f} ms', styles["BodyText"]),
        Spacer(1, 0.2*cm),
        Paragraph("3. Test d’une entrée incorrecte", styles["Heading2"]),
        Paragraph(f'Requête envoyée avec 3 pixels. Code HTTP obtenu : {bad_result["status_code"]}. '
                  f'Réponse : {bad_result["response"]}', styles["BodyText"]),
        Paragraph("4. Visualisation des images testées", styles["Heading2"]),
    ]
    image_path = BASE_DIR / "results" / "images_testees.png"
    if image_path.exists():
        story.append(Image(str(image_path), width=17*cm, height=14.2*cm))
    story.append(PageBreak())
    story.append(Paragraph("5. Réponses aux questions d’analyse", styles["Heading1"]))
    for q, a in QUESTIONS:
        story += [Paragraph(q, styles["Heading3"]), Paragraph(a, styles["BodyText"]), Spacer(1, 0.12*cm)]
    story += [
        Paragraph("6. Conclusion", styles["Heading2"]),
        Paragraph("Ce TP a permis de déployer le modèle Fashion-MNIST sous la forme d’une API FastAPI. "
                  "Le service reçoit une image représentée par 784 pixels, vérifie et prépare les données, "
                  "puis exécute le modèle pré-entraîné. La fonction Softmax permet d’obtenir un score de "
                  "confiance pour chacune des dix classes. Les tests sur vingt images permettent de mesurer "
                  "le taux de réussite et la latence moyenne. Le test avec trois pixels vérifie également "
                  "que l’API refuse une entrée non conforme. Les résultats numériques présentés dans ce "
                  "rapport sont générés automatiquement à partir de l’exécution réelle de test_api.py.",
                  styles["BodyText"]),
        Spacer(1, 0.3*cm),
        Paragraph("<b>Captures à ajouter :</b> placer swagger.png, health.png et prediction.png dans le dossier captures/.",
                  styles["BodyText"]),
    ]
    doc = SimpleDocTemplate(str(BASE_DIR / "compte_rendu.pdf"), pagesize=A4,
                            rightMargin=1.5*cm, leftMargin=1.5*cm,
                            topMargin=1.5*cm, bottomMargin=1.5*cm)
    doc.build(story)
