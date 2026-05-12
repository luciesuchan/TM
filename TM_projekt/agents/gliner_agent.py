from gliner import GLiNER

class GLiNERAgent:
    def __init__(self):
        self.model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
        self.labels = ["person", "address", "phone number", "email", "bank account", "id number", "job title"]
        self.label_mapping = {
            "person": "NAME", "job title": "NAME", "address": "LOC", 
            "email": "CONTACT", "id number": "ID", "phone number": "CONTACT"
        }
        # Rozšířený blacklist o slova, která často ničí Precision
        self.blacklist = [
            "Email", "Telefon", "Adresa", "IČO", "DIČ", "Applications", "Solutions", 
            "Quality", "Systems", "Group", "Management", "Business", "Department"
        ]

    def find_pii(self, text: str):
        findings = []
        try:
            entities = self.model.predict_entities(text, self.labels, threshold=0.38)
            for ent in entities:
                ent_text = ent["text"].strip()
                
                # Precision Boost: Ignorujeme jména/pozice, která nezačínají velkým písmenem 
                # (pokud to není začátek věty), protože to jsou často obecná podstatná jména.
                if ent["label"] in ["person", "job title"] and not ent_text[0].isupper():
                    continue

                if any(b.lower() == ent_text.lower() for b in self.blacklist):
                    continue
                
                findings.append({
                    "start": ent["start"], "end": ent["end"],
                    "label": self.label_mapping.get(ent["label"], "MISC"),
                    "source": "GLiNERAgent"
                })
        except Exception: pass
        return findings