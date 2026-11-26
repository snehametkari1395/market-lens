"""Interface translations and the language instruction for the assistant.

The research assistant answers in the chosen language; SEC filings stay in English,
so filing searches are always run in English.
"""

from __future__ import annotations

LANGUAGES: dict[str, str] = {
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
    "hi": "हिन्दी",
}

# Language names in English, for the model instruction.
LANGUAGE_NAMES_EN = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "hi": "Hindi"}

STRINGS: dict[str, dict[str, str]] = {
    "page_title": {
        "en": "Public-company research, with sources",
        "es": "Investigación de empresas cotizadas, con fuentes",
        "fr": "Recherche sur les sociétés cotées, avec sources",
        "de": "Recherche zu börsennotierten Unternehmen, mit Quellen",
        "pt": "Pesquisa de empresas de capital aberto, com fontes",
        "hi": "सार्वजनिक कंपनियों पर शोध, स्रोतों के साथ",
    },
    "page_intro": {
        "en": "Compare reported financials, explore SEC filing evidence, and ask questions. Every result should be checked against its cited source.",
        "es": "Compara datos financieros reportados, explora la evidencia de los informes ante la SEC y haz preguntas. Verifica cada resultado con su fuente citada.",
        "fr": "Comparez les données financières publiées, explorez les documents déposés auprès de la SEC et posez vos questions. Vérifiez chaque résultat auprès de sa source citée.",
        "de": "Vergleichen Sie gemeldete Finanzdaten, durchsuchen Sie SEC-Berichte und stellen Sie Fragen. Prüfen Sie jedes Ergebnis anhand der zitierten Quelle.",
        "pt": "Compare dados financeiros reportados, explore os documentos enviados à SEC e faça perguntas. Confira cada resultado com a fonte citada.",
        "hi": "रिपोर्ट किए गए वित्तीय आंकड़ों की तुलना करें, SEC फ़ाइलिंग देखें और सवाल पूछें। हर परिणाम को उसके दिए गए स्रोत से जाँचें।",
    },
    "explore_heading": {
        "en": "Explore the research universe", "es": "Explora el universo de investigación", "fr": "Explorer l'univers de recherche",
        "de": "Das Recherche-Universum erkunden", "pt": "Explore o universo de pesquisa", "hi": "शोध के दायरे को देखें",
    },
    "explore_caption": {
        "en": "Choose a company group to add it to the comparison.", "es": "Elige un grupo de empresas para añadirlo a la comparación.",
        "fr": "Choisissez un groupe de sociétés à ajouter à la comparaison.", "de": "Wählen Sie eine Unternehmensgruppe für den Vergleich.",
        "pt": "Escolha um grupo de empresas para adicioná-lo à comparação.", "hi": "तुलना में जोड़ने के लिए कंपनियों का समूह चुनें।",
    },
    "explore_button": {
        "en": "Explore companies", "es": "Ver empresas", "fr": "Voir les sociétés", "de": "Unternehmen ansehen",
        "pt": "Ver empresas", "hi": "कंपनियाँ देखें",
    },
    "tab_overview": {"en": "Company overview", "es": "Resumen", "fr": "Aperçu", "de": "Überblick", "pt": "Visão geral", "hi": "कंपनी अवलोकन"},
    "tab_analytics": {"en": "Analytics", "es": "Análisis", "fr": "Analyses", "de": "Analysen", "pt": "Análises", "hi": "विश्लेषण"},
    "tab_events": {"en": "Company events", "es": "Eventos", "fr": "Événements", "de": "Ereignisse", "pt": "Eventos", "hi": "कंपनी घटनाएँ"},
    "starter_heading": {"en": "Not sure where to start? Try one:", "es": "¿No sabes por dónde empezar? Prueba una:", "fr": "Vous ne savez pas par où commencer ? Essayez :", "de": "Keine Idee, wo anfangen? Probieren Sie:", "pt": "Não sabe por onde começar? Experimente:", "hi": "कहाँ से शुरू करें? इनमें से एक आज़माएँ:"},
    "tab_valuation": {"en": "Valuation", "es": "Valoración", "fr": "Valorisation", "de": "Bewertung", "pt": "Avaliação", "hi": "मूल्यांकन"},
    "tab_calendar": {"en": "Calendar", "es": "Calendario", "fr": "Calendrier", "de": "Kalender", "pt": "Calendário", "hi": "कैलेंडर"},
    "tab_watchlist": {"en": "Watchlist", "es": "Seguimiento", "fr": "Suivi", "de": "Beobachtungsliste", "pt": "Acompanhamento", "hi": "वॉचलिस्ट"},
    "tab_private": {"en": "Private companies", "es": "Empresas privadas", "fr": "Sociétés privées", "de": "Private Firmen", "pt": "Empresas privadas", "hi": "निजी कंपनियाँ"},
    "tab_shortlist": {"en": "Research shortlist", "es": "Lista corta", "fr": "Présélection", "de": "Auswahlliste", "pt": "Lista curta", "hi": "शॉर्टलिस्ट"},
    "tab_assistant": {"en": "Research assistant", "es": "Asistente", "fr": "Assistant", "de": "Assistent", "pt": "Assistente", "hi": "शोध सहायक"},
    "tab_filings": {"en": "Filing library", "es": "Biblioteca de informes", "fr": "Bibliothèque de documents", "de": "Berichtsbibliothek", "pt": "Biblioteca de relatórios", "hi": "फ़ाइलिंग लाइब्रेरी"},
    "tab_evaluation": {"en": "RAG evaluation", "es": "Evaluación RAG", "fr": "Évaluation RAG", "de": "RAG-Bewertung", "pt": "Avaliação RAG", "hi": "RAG मूल्यांकन"},
    "tab_monitoring": {"en": "Monitoring", "es": "Monitoreo", "fr": "Supervision", "de": "Monitoring", "pt": "Monitoramento", "hi": "निगरानी"},
    "tab_guide": {"en": "Help & guide", "es": "Ayuda", "fr": "Aide", "de": "Hilfe", "pt": "Ajuda", "hi": "सहायता"},
    "language": {"en": "Language", "es": "Idioma", "fr": "Langue", "de": "Sprache", "pt": "Idioma", "hi": "भाषा"},
    "model_heading": {"en": "Model", "es": "Modelo", "fr": "Modèle", "de": "Modell", "pt": "Modelo", "hi": "मॉडल"},
    "provider": {"en": "Provider", "es": "Proveedor", "fr": "Fournisseur", "de": "Anbieter", "pt": "Provedor", "hi": "प्रदाता"},
    "sec_email": {"en": "SEC contact email", "es": "Email de contacto para la SEC", "fr": "E-mail de contact SEC", "de": "SEC-Kontakt-E-Mail", "pt": "E-mail de contato da SEC", "hi": "SEC संपर्क ईमेल"},
    "history": {"en": "Financial history", "es": "Historial financiero", "fr": "Historique financier", "de": "Finanzhistorie", "pt": "Histórico financeiro", "hi": "वित्तीय इतिहास"},
    "companies": {"en": "Companies", "es": "Empresas", "fr": "Sociétés", "de": "Unternehmen", "pt": "Empresas", "hi": "कंपनियाँ"},
    "search_method": {"en": "Filing search method", "es": "Método de búsqueda", "fr": "Méthode de recherche", "de": "Suchmethode", "pt": "Método de busca", "hi": "खोज का तरीका"},
    "save_defaults": {"en": "Save these as my defaults", "es": "Guardar como mis preferencias", "fr": "Enregistrer comme préférences", "de": "Als Standard speichern", "pt": "Salvar como meu padrão", "hi": "इन्हें मेरी डिफ़ॉल्ट सेटिंग बनाएँ"},
    "sign_out": {"en": "Sign out", "es": "Cerrar sesión", "fr": "Se déconnecter", "de": "Abmelden", "pt": "Sair", "hi": "साइन आउट"},
    "compare_heading": {"en": "Compare fundamentals", "es": "Comparar fundamentales", "fr": "Comparer les fondamentaux", "de": "Kennzahlen vergleichen", "pt": "Comparar fundamentos", "hi": "मूल बातों की तुलना करें"},
    "metric": {"en": "Financial metric", "es": "Métrica financiera", "fr": "Indicateur financier", "de": "Finanzkennzahl", "pt": "Métrica financeira", "hi": "वित्तीय मापदंड"},
    "load_sec": {"en": "Load SEC data", "es": "Cargar datos de la SEC", "fr": "Charger les données SEC", "de": "SEC-Daten laden", "pt": "Carregar dados da SEC", "hi": "SEC डेटा लोड करें"},
    "refresh_sec": {"en": "Refresh from SEC now", "es": "Actualizar desde la SEC", "fr": "Actualiser depuis la SEC", "de": "Jetzt von der SEC aktualisieren", "pt": "Atualizar da SEC agora", "hi": "अभी SEC से ताज़ा करें"},
    "assistant_heading": {"en": "Ask the filings and financial data", "es": "Pregunta a los informes y datos financieros", "fr": "Interrogez les documents et données financières", "de": "Fragen an Berichte und Finanzdaten", "pt": "Pergunte aos relatórios e dados financeiros", "hi": "फ़ाइलिंग और वित्तीय डेटा से पूछें"},
    "assistant_hint": {
        "en": "Try: “Compare Apple and Microsoft revenue growth” or “What risks does Microsoft's latest 10-K discuss?”",
        "es": "Prueba: «Compara el crecimiento de ingresos de Apple y Microsoft» o «¿Qué riesgos menciona el último 10-K de Microsoft?»",
        "fr": "Essayez : « Comparez la croissance du chiffre d'affaires d'Apple et de Microsoft » ou « Quels risques le dernier 10-K de Microsoft mentionne-t-il ? »",
        "de": "Beispiel: „Vergleiche das Umsatzwachstum von Apple und Microsoft“ oder „Welche Risiken nennt Microsofts letzter 10-K?“",
        "pt": "Experimente: “Compare o crescimento da receita da Apple e da Microsoft” ou “Quais riscos o último 10-K da Microsoft menciona?”",
        "hi": "आज़माएँ: “Apple और Microsoft की राजस्व वृद्धि की तुलना करें” या “Microsoft की नवीनतम 10-K में कौन-से जोखिम बताए गए हैं?”",
    },
    "chat_placeholder": {"en": "Ask about company financials or SEC filings", "es": "Pregunta sobre finanzas o informes de la SEC", "fr": "Question sur les finances ou les documents SEC", "de": "Frage zu Finanzen oder SEC-Berichten", "pt": "Pergunte sobre finanças ou relatórios da SEC", "hi": "कंपनी के वित्त या SEC फ़ाइलिंग के बारे में पूछें"},
    "export": {"en": "Export", "es": "Exportar", "fr": "Exporter", "de": "Exportieren", "pt": "Exportar", "hi": "निर्यात"},
    "clear_chat": {"en": "Clear conversation", "es": "Borrar conversación", "fr": "Effacer la conversation", "de": "Unterhaltung löschen", "pt": "Limpar conversa", "hi": "बातचीत साफ़ करें"},
    "sources": {"en": "Sources", "es": "Fuentes", "fr": "Sources", "de": "Quellen", "pt": "Fontes", "hi": "स्रोत"},
    "was_helpful": {"en": "Was this answer helpful?", "es": "¿Te fue útil esta respuesta?", "fr": "Cette réponse vous a-t-elle aidé ?", "de": "War diese Antwort hilfreich?", "pt": "Esta resposta foi útil?", "hi": "क्या यह उत्तर उपयोगी था?"},
    "disclaimer": {
        "en": "SEC-reported figures are historical disclosures, not predictions. Market prices use Alpha Vantage when configured and may be delayed. This app does not provide personalized investment advice.",
        "es": "Las cifras reportadas a la SEC son datos históricos, no predicciones. Los precios de mercado provienen de Alpha Vantage y pueden tener retraso. Esta aplicación no ofrece asesoramiento de inversión personalizado.",
        "fr": "Les chiffres déclarés à la SEC sont des données historiques, pas des prévisions. Les cours proviennent d'Alpha Vantage et peuvent être différés. Cette application ne fournit pas de conseil en investissement personnalisé.",
        "de": "Bei der SEC gemeldete Zahlen sind historische Angaben, keine Prognosen. Kurse stammen von Alpha Vantage und können verzögert sein. Diese App bietet keine persönliche Anlageberatung.",
        "pt": "Os números reportados à SEC são dados históricos, não previsões. Os preços vêm da Alpha Vantage e podem ter atraso. Este aplicativo não oferece aconselhamento de investimento personalizado.",
        "hi": "SEC में दर्ज आंकड़े ऐतिहासिक जानकारी हैं, भविष्यवाणी नहीं। बाज़ार मूल्य Alpha Vantage से आते हैं और देरी से हो सकते हैं। यह ऐप व्यक्तिगत निवेश सलाह नहीं देता।",
    },
    "research_only": {"en": "Research only · Not investment advice", "es": "Solo investigación · No es asesoramiento", "fr": "Recherche uniquement · Pas un conseil", "de": "Nur Recherche · Keine Anlageberatung", "pt": "Apenas pesquisa · Não é recomendação", "hi": "केवल शोध · निवेश सलाह नहीं"},
}


def translate(key: str, language: str = "en") -> str:
    """The string in the chosen language, falling back to English, then to the key itself."""
    entry = STRINGS.get(key)
    if not entry:
        return key
    return entry.get(language) or entry.get("en") or key


def language_instruction(language: str) -> str:
    """Extra system-prompt text so the assistant answers in the user's language."""
    if language == "en" or language not in LANGUAGE_NAMES_EN:
        return ""
    name = LANGUAGE_NAMES_EN[language]
    return (
        f"\nWrite your final answer in {name}, whatever language the question is in. Keep company names, "
        "tickers, numbers, units, fiscal years, SEC item names and URLs exactly as they appear in the sources. "
        "SEC filings are in English: always write the question for search_filing_knowledge in English."
    )
