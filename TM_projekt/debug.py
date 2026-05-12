"""
Debug skript - ukáže přesně co model predikuje.
Spusť: python debug_model.py
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import Counter

MODEL_PATH = "./model_final_pro"
TEXT = "A student's assessment was found on device bearing IMEI: 06-184755-866851-3."

print("=" * 60)
print(f"Model: {MODEL_PATH}")
print(f"Text:  {TEXT}")
print("=" * 60)

# --- 1. Info o modelu ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

print(f"\n📊 Model info:")
print(f"   num_labels: {model.config.num_labels}")
print(f"   id2label (prvních 10): {dict(list(model.config.id2label.items())[:10])}")

# --- 2. Raw logits pro každý token ---
inputs = tokenizer(TEXT, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits[0]
probs  = torch.softmax(logits, dim=-1)
pred_ids = logits.argmax(dim=-1)

print(f"\n🔍 Token-level predikce:")
print(f"{'TOKEN':<25} {'LABEL':<20} {'CONF':>6}")
print("-" * 55)
for tok, pid in zip(tokens, pred_ids):
    label = model.config.id2label[pid.item()]
    conf  = probs[pred_ids.tolist().index(pid.item()) if False else list(range(len(pred_ids))).index(list(pred_ids).index(pid))].max().item()
    conf  = probs[tokens.index(tok) if tok in tokens else 0, pid].item()
    print(f"{tok:<25} {label:<20} {conf:>6.3f}")

print(f"\n📈 Distribuce predikcí:")
counts = Counter(model.config.id2label[p.item()] for p in pred_ids)
for label, cnt in counts.most_common():
    print(f"   {label}: {cnt}x")

# --- 3. Pipeline test ---
print(f"\n🔧 Pipeline test (aggregation_strategy=simple):")
ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
results = ner(TEXT)
if results:
    for r in results:
        print(f"   {r['entity_group']:20} '{TEXT[r['start']:r['end']]}' (score={r['score']:.3f})")
else:
    print("   Žádné entity nenalezeny (model predikuje jen O)")

# --- 4. Klíčová otázka: je O vůbec v id2label? ---
print(f"\n❓ Je 'O' v id2label?", "O" in model.config.id2label.values())
o_ids = [k for k,v in model.config.id2label.items() if v == "O"]
print(f"   O má id: {o_ids}")
print(f"   Nejčastěji predikovaný label: {counts.most_common(1)[0]}")