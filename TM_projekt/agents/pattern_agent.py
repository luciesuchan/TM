import re

class PatternAgent:
    def __init__(self):
        self.rules = {
            # --- KONTAKTNÍ ÚDAJE ---
            "CONTACT": [
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email
                r'(\+420)?\s?([1-9][0-9]{2})\s?([0-9]{3})\s?([0-9]{3})', # CZ telefon
                r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+' # URL
            ],

            # --- FINANČNÍ ÚDAJE ---
            "FINANCE": [
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', # Karty
                r'[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}', # IBAN
                r'\b\d{2,10}/\d{4}\b', # Číslo účtu CZ
                r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b' # Bitcoin
            ],

            # --- IDENTIFIKÁTORY ---
            "ID": [
                r'\b\d{2}-\d{6}-\d{6}-\d{1}\b', # IMEI
                r'\b\d{6}/\d{3,4}\b',           # Rodné číslo
                r'\b\d{8}\b',                   # IČO
                r'CZ\d{8,10}',                  # DIČ
                r'\b[A-Z0-9]{7,9}\b',           # OP / Pas
                r'\b[a-fA-F0-9]{2}(?::[a-fA-F0-9]{2}){5}\b' # MAC adresa
            ],

            "LOC": [
                r'\b\d{3}\s?\d{2}\b', # PSČ
                r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', # IPv4
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b' # IPv6
            ]
        }

    def find_pii(self, text: str):
        findings = []
        for label, patterns in self.rules.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    findings.append({
                        "start": match.start(),
                        "end": match.end(),
                        "label": label,
                        "source": "PatternAgent"
                    })
        return findings