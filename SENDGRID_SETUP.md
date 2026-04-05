# 📧 Configuration SendGrid pour RetainIQ

## Vue d'ensemble

RetainIQ envoie des **rapports PDF hebdomadaires** à chaque utilisateur via **SendGrid** (service d'email fiable).

**Si SendGrid n'est pas configuré**, les rapports sont **sauvegardés localement** dans le dossier `reports_archive/`.

---

## 📋 Étape 1 : Créer un compte SendGrid

1. **Allez sur** [sendgrid.com](https://sendgrid.com)
2. **Créez un compte gratuit** (100 emails/jour inclus)
3. **Vérifiez votre email** pour activer le compte
4. **Connectez-vous** à votre dashboard

---

## 🔑 Étape 2 : Générer une clé API

1. Dans le dashboard, allez à **Settings → API Keys**
2. Cliquez sur **Create API Key**
3. Donnez un nom : `RetainIQ`
4. Choisissez **Restricted Access**
5. Sélectionnez **Mail Send** → **Full Access**
6. Cliquez **Create & Copy**
7. **Copiez la clé** (elle commence par `SG.`)

⚠️ **Important** : C'est la seule fois où vous verrez la clé. Gardez-la en sécurité !

---

## 📧 Étape 3 : Configurer l'adresse émetteur

SendGrid vous demande de **vérifier l'adresse émetteur** avant d'envoyer.

### Option A : Vérifier un email simple

1. Dans le dashboard, allez à **Settings → Sender Authentication**
2. Cliquez **Verify a Single Sender**
3. Entrez :
   - **From Email** : `noreply@votreentreprise.com` (ou votre email)
   - **From Name** : `RetainIQ`
4. Vérifiez l'email dans votre boîte de réception
5. ✅ Activé !

### Option B : Configurer un domaine (recommandé en prod)

Voir [documentation SendGrid](https://sendgrid.com/docs/glossary/domain-authentication/) pour authentifier votre propre domaine.

---

## ⚙️ Étape 4 : Configurer les variables d'environnement

Éditez le fichier `.env` à la racine du projet :

```env
# SendGrid Configuration
SENDGRID_API_KEY=SG.votre_clé_ici_pas_de_guillemets
SENDER_EMAIL=noreply@votreentreprise.com
SENDER_NAME=RetainIQ
```

### Exemple :

```env
SENDGRID_API_KEY=SG.6L8kW9pQxYz2A4mN6B3t5R1k9W8...
SENDER_EMAIL=reports@orange.com
SENDER_NAME=Orange Retention
```

---

## 🧪 Étape 5 : Tester la configuration

### Test 1 : Vérifier les imports

```bash
python -c "from weekly_report_job import send_weekly_reports; print('[OK] Modules chargés')"
```

**Résultat attendu** : `[OK] Modules chargés`

### Test 2 : Générer un rapport de test

```bash
python -c "
import pandas as pd
from email_reports import generate_pdf_report

# Créer des données de test
df = pd.DataFrame({
    'tenure': [12, 24, 6],
    'MonthlyCharges': [65.5, 89.0, 45.0],
    'TotalCharges': [786, 2136, 270],
    'Churn': [0, 1, 0],
    'ChurnProba': [0.25, 0.85, 0.15]
})

# Générer le PDF
generate_pdf_report(
    df=df,
    company_name='Test Company',
    sector='Telecom',
    output_path='test_report.pdf'
)
print('[OK] PDF généré: test_report.pdf')
"
```

**Résultat attendu** : Fichier `test_report.pdf` créé

### Test 3 : Tester un envoi email (optionnel)

```bash
python -c "
from email_reports import send_pdf_via_sendgrid
import os

success, message = send_pdf_via_sendgrid(
    to_email='votre-email@example.com',
    subject='Test RetainIQ',
    body_text='Ceci est un test',
    pdf_path='test_report.pdf',
    from_email=os.getenv('SENDER_EMAIL'),
    from_name=os.getenv('SENDER_NAME')
)
print(f'Résultat: {message}')
"
```

**Résultats possibles** :
- ✅ `Email envoyé à votre-email@example.com`
- ⚠️ `[FALLBACK] SendGrid non configuré → PDF sauvegardé localement...`
- ⚠️ `[FALLBACK] SendGrid a échoué...`

---

## 📁 Dossier des rapports

Si SendGrid n'est pas configuré ou échoue, les PDFs sont sauvegardés ici :

```
reports_archive/
├── report_orange@orange_com_20250405_143025.pdf
├── report_fitnesspark@fit_com_20250405_143030.pdf
└── ...
```

Vous pouvez les télécharger manuellement.

---

## 🚀 Activer les rapports hebdomadaires

Une fois configuré, les rapports s'envoient automatiquement chaque semaine.

1. **Lancez le dashboard** : `streamlit run churn_prediction_dashboard.py`
2. **Allez à** → ⏰ Rapports Planifiés
3. **Configurez le jour/heure** (exemple : Lundi 09:00)
4. Les rapports s'envoient automatiquement chaque semaine

---

## ❓ Dépannage

### "SENDGRID_API_KEY manquante"

✅ **Solution** : Vérifiez que `.env` existe et contient `SENDGRID_API_KEY=SG.xxx`

### "Email non vérifié"

✅ **Solution** : Allez à Settings → Sender Authentication et vérifiez votre adresse email

### "Les rapports ne s'envoient pas"

✅ **Solutions** :
1. Vérifiez que vous avez uploaddé vos données (page 📤)
2. Vérifiez les logs : `reports_archive/` contient-il des PDFs ?
3. Testez la connexion SendGrid (Étape 5 ci-dessus)
4. Vérifiez les logs de l'application Streamlit

### "SendGrid refusé l'email"

✅ **Solutions** :
1. Vérifiez que `SENDER_EMAIL` est **vérifiée** dans SendGrid
2. Vérifiez que l'adresse destinataire est **valide**
3. Vérifiez les quotas SendGrid (100 emails/jour en gratuit)

---

## 📊 Quota gratuit SendGrid

- ✅ **100 emails/jour** inclus gratuitement
- ✅ **Support email** 24/7
- ❌ Après 100/jour, vous devez passer en plan payant

---

## 🔐 Sécurité

- ✅ Ne commitez **jamais** le `.env` dans Git (déjà dans `.gitignore`)
- ✅ Gardez votre clé API **secrète**
- ✅ Si vous avez leaké la clé, supprimez-la dans SendGrid et créez-en une nouvelle

---

## 📚 Ressources

- [SendGrid Documentation](https://sendgrid.com/docs/)
- [API Python SendGrid](https://github.com/sendgrid/sendgrid-python)
- [Vérifier un émetteur](https://sendgrid.com/docs/glossary/sender-identity/)

