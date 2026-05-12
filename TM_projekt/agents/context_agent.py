"""import os
import torch
from transformers import pipeline


class ContextAgent:
    def __init__(self, model_path: str = "./moje_pii_model"):
        abs_path = os.path.abspath(model_path)

        if not os.path.isdir(abs_path):
            print(f"[ContextAgent] ⚠️  Model nenalezen v: {abs_path}")
            print("[ContextAgent]    Spusť nejprve train.py pro natrénování modelu.")
            self.ner_model = None
            return

        # MPS = Apple Silicon GPU, jinak CPU
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"[ContextAgent] Načítám model z: {abs_path}  (device={device})")
        self.ner_model = pipeline(
            "ner",
            model=abs_path,
            aggregation_strategy="simple",   # B-/I- tokeny se sloučí automaticky
            device=device,
        )
        print("[ContextAgent] ✅ Model načten.")

    def find_pii(self, text: str) -> list[dict]:
        if self.ner_model is None:
            return []

        try:
            results = self.ner_model(text)
        except Exception as e:
            print(f"[ContextAgent] ❌ Chyba inference: {e}")
            return []

        findings = []
        for r in results:
            label = r["entity_group"]
            # Přeskočíme token "O" (není PII)
            if label in ("O", ""):
                continue
            # Odstraníme B-/I- prefix, pokud model nepoužívá aggregation správně
            label = label.replace("B-", "").replace("I-", "")
            findings.append({
                "start":  r["start"],
                "end":    r["end"],
                "label":  label,
                "source": "ContextAgent",
            })
        return findings"""
"""from transformers import pipeline
import torch
import os

class ContextAgent:
    def __init__(self, model_path):
        abs_path = os.path.abspath(model_path)
        # MPS pro zrychlení na Macu
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.ner_model = pipeline("ner", model=abs_path, aggregation_strategy="simple", device=device)

    def find_pii(self, text: str):
        try:
            results = self.ner_model(text)
            findings = []
            for r in results:
                if r['score'] < 0.15: continue
                
                start, end = r['start'], r['end']
                while start > 0 and text[start-1] not in [" ", "\n", "\t", ".", ",", "!", "?", ":"]:
                    start -= 1
                while end < len(text) and text[end] not in [" ", "\n", "\t", ".", ",", "!", "?", ":"]:
                    end += 1

                lbl = r['entity_group'].replace("B-", "").replace("I-", "")
                if lbl == "O": continue
                
                findings.append({
                    "start": start, "end": end,
                    "label": lbl, "source": "ContextAgent"
                })
            return findings
        except Exception: return []"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class ContextAgent:
    def __init__(self, model_path="./model_final_pro"):
        print(f"[*] Načítám tvůj vytrénovaný model z {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        self.nlp = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer, 
            aggregation_strategy="simple",
            device="mps" if torch.backends.mps.is_available() else -1
        )
        self.label_map = {"NAME": "NAME", "LOC": "LOC", "CONTACT": "CONTACT", "ID": "ID", "FINANCE": "FINANCE"}

    def find_pii(self, text: str):
        if not text or not text.strip(): return []
        
        chunk_size = 1000 
        overlap = 200      
        findings = []
        seen_spans = set() 

        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if not chunk: break
            
            results = self.nlp(chunk)
            
            for res in results:
                if res['score'] > 0.30:
                    clean_label = res['entity_group'].replace("B-", "").replace("I-", "")
                    
                    real_start = res['start'] + i
                    real_end = res['end'] + i
                    
                    span_id = (real_start, real_end, clean_label)
                    if span_id not in seen_spans:
                        findings.append({
                            "start": real_start,
                            "end": real_end,
                            "label": self.label_map.get(clean_label, "MISC"),
                            "source": "ContextAgent",
                            "score": res['score']
                        })
                        seen_spans.add(span_id)
        
        return findings