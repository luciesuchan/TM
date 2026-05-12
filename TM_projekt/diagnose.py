"""
Ověří přesně co jde do modelu před tréninkem.
Spusť: python diagnose3.py
"""
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "microsoft/deberta-v3-small"
raw_ds = load_dataset("ai4privacy/pii-masking-200k", split="train")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Sbíráme labely
unique_labels = set()
for row in raw_ds.select(range(100)):
    unique_labels.update(row["mbert_bio_labels"])
sorted_labels = sorted(unique_labels)
if "O" not in sorted_labels:
    sorted_labels = ["O"] + sorted_labels
label2id = {lbl: i for i, lbl in enumerate(sorted_labels)}
o_id = label2id["O"]

ex = raw_ds[0]
print("=== PŘÍKLAD 0 ===")
print(f"source_text: {ex['source_text']}")
print(f"\nmbert_text_tokens ({len(ex['mbert_text_tokens'])}): {ex['mbert_text_tokens']}")
print(f"\nmbert_bio_labels  ({len(ex['mbert_bio_labels'])}): {ex['mbert_bio_labels']}")

# Tokenizujeme jako v train.py
word_list = [tok.lstrip("##") if tok.startswith("##") else tok
             for tok in ex["mbert_text_tokens"]]
print(f"\nPo lstrip ## ({len(word_list)}): {word_list}")

enc = tokenizer(word_list, is_split_into_words=True, truncation=True, max_length=128)
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])
word_ids = enc.word_ids()

print(f"\nDeberta tokeny ({len(tokens)}): {tokens}")
print(f"word_ids: {word_ids}")

# Zarovnání
str_labels = ex["mbert_bio_labels"]
label_ids = []
prev = None
for wid in word_ids:
    if wid is None:
        label_ids.append(-100)
    elif wid != prev:
        lbl = str_labels[wid] if wid < len(str_labels) else "O"
        label_ids.append(label2id.get(lbl, o_id))
    else:
        label_ids.append(-100)
    prev = wid

print(f"\nLabel IDs: {label_ids}")
active = sum(1 for l in label_ids if l != -100)
pii    = sum(1 for l in label_ids if l not in (-100, o_id))
print(f"\nAktivní tokeny: {active}/{len(label_ids)}, PII tokeny: {pii}")

# Ukažme zarovnání token↔label
print("\n=== TOKEN ↔ LABEL ZAROVNÁNÍ ===")
for tok, wid, lid in zip(tokens, word_ids, label_ids):
    if lid == -100:
        lbl_str = "[ignored]"
    else:
        lbl_str = sorted_labels[lid]
    orig = str_labels[wid] if wid is not None and wid < len(str_labels) else "-"
    marker = " ← PII!" if lid not in (-100, o_id) else ""
    print(f"  {tok:<20} wid={str(wid):<4} orig={orig:<15} → {lbl_str}{marker}")

# Zkontrolujeme jestli vocab_size souhlasí
print(f"\n=== VOCAB CHECK ===")
print(f"Počet labelů: {len(sorted_labels)}")
max_label_id = max(label2id.values())
print(f"Max label id: {max_label_id}")
print(f"Jsou labely v rozumném rozsahu? {max_label_id < 200}")