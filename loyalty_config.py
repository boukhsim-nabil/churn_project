# ═══════════════════════════════════════════════════════════════════════════════
# loyalty_config.py — Catalogue de récompenses par secteur
# RetainIQ · Module Fidélité & Rétention · Marché Marocain
# ═══════════════════════════════════════════════════════════════════════════════

# ── CATALOGUE COMPLET PAR SECTEUR ───────────────────────────────────────────
REWARDS_CATALOG = {

    "📱 Télécom": {
        "sauvetage": [
            "Pass Internet 2Go offert (valable 7 jours)",
            "Pass Internet 5Go offert (valable 15 jours)",
            "Pass Réseaux Sociaux 1 mois gratuit",
            "1 mois Shahid VIP offert",
            "Remise de 20% sur la prochaine facture",
            "Remise de 30% sur la prochaine facture",
            "Gel de l'abonnement 1 mois sans frais",
            "Appels illimités vers tous les opérateurs 48h",
        ],
        "fidelite": [
            "Double des points Club (mois en cours)",
            "Triple des points Club (mois en cours)",
            "Recharge bonus x2 sur la prochaine recharge",
            "Changement de carte SIM gratuit",
            "Surclassement vers offre supérieure 1 mois",
            "Accès prioritaire au service client dédié",
            "Invitation événement VIP Maroc Telecom / IAM",
        ],
        "seuil_sauvetage": 0.50,
        "seuil_fidelite":  0.35,
        "tenure_fidelite": 18,  # mois minimum pour cohorte fidélité
        "devise": "MAD",
    },

    "💪 Salle de Sport": {
        "sauvetage": [
            "Gel de l'abonnement 1 mois sans perte",
            "Gel de l'abonnement 2 mois sans perte",
            "Accès multi-clubs (toutes les salles du réseau) 1 mois",
            "1 séance coaching personnalisé offerte",
            "2 séances coaching personnalisé offertes",
            "Abonnement nutrition 1 mois offert",
            "Suspension puis réactivation sans frais",
        ],
        "fidelite": [
            "Pass invité 2 semaines (ami ou famille)",
            "Goodies exclusifs du club (tote bag, shaker, serviette)",
            "Accès zone premium / spa 1 mois",
            "Cours collectifs illimités 1 mois",
            "Réduction -20% sur le renouvellement annuel",
            "Photo professionnelle transformation offerte",
            "Invitation soirée membres fidèles",
        ],
        "seuil_sauvetage": 0.50,
        "seuil_fidelite":  0.35,
        "tenure_fidelite": 12,
        "devise": "MAD",
    },

    "🛍️ E-commerce": {
        "sauvetage": [
            "Livraison express gratuite sur la prochaine commande",
            "Promotion -15% valable 48h (code exclusif)",
            "Promotion -20% valable 48h (code exclusif)",
            "Cadeau surprise ajouté au prochain panier",
            "Remboursement partiel sur dernière commande (-10%)",
            "Points fidélité x3 sur la prochaine commande",
            "Accès vente privée flash 24h",
        ],
        "fidelite": [
            "Accès anticipé aux soldes (48h avant le public)",
            "Livraison gratuite sur 3 prochaines commandes",
            "Cadeau personnalisé pour anniversaire client",
            "Badge VIP affiché sur le compte",
            "Service client prioritaire via WhatsApp",
            "Invitation bêta test nouveaux produits",
            "Cashback 5% sur commandes du trimestre",
        ],
        "seuil_sauvetage": 0.50,
        "seuil_fidelite":  0.35,
        "tenure_fidelite": 6,
        "devise": "MAD",
    },

    "🎓 EdTech": {
        "sauvetage": [
            "Gel du compte 1 mois sans perte de progression",
            "Accès illimité à toutes les formations 1 mois",
            "Session de coaching 1h avec un formateur",
            "Certificat de progression envoyé par email",
            "Réduction -25% sur le renouvellement",
            "Accès formations premium 2 semaines offert",
        ],
        "fidelite": [
            "Certificat d'excellence mensuel personnalisé",
            "Accès early bird aux nouveaux modules",
            "Mention dans la newsletter communauté",
            "1 formation supplémentaire offerte",
            "Badge 'Apprenant fidèle' sur le profil",
            "Invitation webinaire exclusif membres fidèles",
        ],
        "seuil_sauvetage": 0.50,
        "seuil_fidelite":  0.35,
        "tenure_fidelite": 6,
        "devise": "MAD",
    },

    "☁️ SaaS B2B": {
        "sauvetage": [
            "Gel du tarif actuel pour 12 mois supplémentaires",
            "Audit complet du compte offert (2h consultant)",
            "Accès fonctionnalités Premium 2 mois gratuit",
            "Migration de plan sans frais de changement",
            "Formation équipe onsite 1 journée offerte",
            "Crédits API supplémentaires (x2 quota mensuel)",
        ],
        "fidelite": [
            "Étude de cas client publiée sur notre site",
            "2 licences utilisateurs supplémentaires gratuites",
            "Support WhatsApp direct (ligne dédiée)",
            "Co-marketing : mention dans nos réseaux sociaux",
            "Accès bêta nouvelles fonctionnalités",
            "Rapport d'usage mensuel personnalisé",
            "Invitation conférence annuelle SaaS Maroc",
        ],
        "seuil_sauvetage": 0.50,
        "seuil_fidelite":  0.35,
        "tenure_fidelite": 12,
        "devise": "MAD",
    },
}

# ── MESSAGES DE GRATITUDE PAR SECTEUR ───────────────────────────────────────
GRATITUDE_MESSAGES = {
    "📱 Télécom": {
        "anniversaire": (
            "Cher(e) client(e),\n\n"
            "Cela fait exactement {annees} an(s) que vous nous faites confiance. "
            "Merci pour votre fidélité exceptionnelle — vous faites partie de nos clients les plus précieux.\n\n"
            "En signe de reconnaissance, nous vous offrons {reward}.\n\n"
            "Merci d'être avec nous depuis {tenure} mois.\n\n"
            "L'équipe RetainIQ Télécom"
        ),
        "mensuel": (
            "Cher(e) client(e),\n\n"
            "Ce mois-ci, nous voulions simplement vous remercier pour votre confiance continue. "
            "Votre satisfaction est notre priorité absolue.\n\n"
            "Vous êtes avec nous depuis {tenure} mois — c'est une relation que nous valorisons profondément.\n\n"
            "Merci et à très bientôt,\nL'équipe RetainIQ Télécom"
        ),
    },
    "💪 Salle de Sport": {
        "anniversaire": (
            "Cher(e) membre,\n\n"
            "Aujourd'hui marque votre {annees}e anniversaire avec nous. "
            "Votre engagement envers votre santé et votre club est une source d'inspiration.\n\n"
            "Pour célébrer ce moment, nous vous offrons {reward}.\n\n"
            "Merci d'être notre membre depuis {tenure} mois. Continuez comme ça !\n\n"
            "Votre équipe Sport & Bien-être"
        ),
        "mensuel": (
            "Cher(e) membre,\n\n"
            "Nous voulions vous féliciter pour votre engagement régulier ce mois-ci. "
            "Chaque séance compte, et vous le prouvez chaque jour.\n\n"
            "Membre depuis {tenure} mois — nous sommes fiers de vous accompagner dans votre parcours.\n\n"
            "À la prochaine séance,\nVotre équipe Sport & Bien-être"
        ),
    },
    "🛍️ E-commerce": {
        "anniversaire": (
            "Cher(e) client(e) fidèle,\n\n"
            "Il y a exactement {annees} an(s), vous avez passé votre première commande chez nous. "
            "Depuis, vous faites partie de notre communauté.\n\n"
            "Pour fêter cet anniversaire, nous vous offrons {reward}.\n\n"
            "Merci pour votre fidélité depuis {tenure} mois.\n\n"
            "Avec toute notre reconnaissance,\nL'équipe E-commerce"
        ),
        "mensuel": (
            "Cher(e) client(e),\n\n"
            "Un simple merci ce mois-ci pour votre confiance. "
            "Chaque commande que vous passez nous permet de nous améliorer.\n\n"
            "Client(e) depuis {tenure} mois — nous sommes heureux de vous compter parmi nos fidèles.\n\n"
            "À très bientôt,\nL'équipe E-commerce"
        ),
    },
    "🎓 EdTech": {
        "anniversaire": (
            "Cher(e) apprenant(e),\n\n"
            "Cela fait {annees} an(s) que vous apprenez avec nous. "
            "Votre curiosité et votre persévérance sont remarquables.\n\n"
            "Pour célébrer cette étape, nous vous offrons {reward}.\n\n"
            "Merci d'apprendre avec nous depuis {tenure} mois.\n\n"
            "Continuez à apprendre,\nL'équipe EdTech"
        ),
        "mensuel": (
            "Cher(e) apprenant(e),\n\n"
            "Ce mois-ci, nous célébrons votre progression. "
            "Chaque module complété vous rapproche de vos objectifs.\n\n"
            "Avec nous depuis {tenure} mois — bravo pour votre régularité.\n\n"
            "L'équipe EdTech"
        ),
    },
    "☁️ SaaS B2B": {
        "anniversaire": (
            "Cher(e) partenaire,\n\n"
            "Cela fait {annees} an(s) que votre entreprise utilise notre plateforme. "
            "Cette relation est l'une des plus précieuses pour nous.\n\n"
            "En signe de gratitude, nous vous offrons {reward}.\n\n"
            "Merci de nous faire confiance depuis {tenure} mois.\n\n"
            "Votre équipe partenaires SaaS"
        ),
        "mensuel": (
            "Cher(e) partenaire,\n\n"
            "Nous voulions prendre un moment pour vous remercier de votre confiance continue. "
            "Votre feedback nous aide à nous améliorer chaque jour.\n\n"
            "Partenaire depuis {tenure} mois — nous sommes fiers de cette collaboration.\n\n"
            "Votre équipe partenaires SaaS"
        ),
    },
}

# ── SEUILS DE SEGMENTATION ───────────────────────────────────────────────────
SEGMENTATION_CONFIG = {
    "champion_proba_max":   0.20,   # ChurnProba < 0.20
    "champion_tenure_min":  12,     # Tenure >= 12 mois
    "sauvetage_proba_min":  0.50,   # ChurnProba > 0.50
    "fidelite_proba_max":   0.35,   # ChurnProba < 0.35
}