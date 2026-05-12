"""import os
import numpy as np
import torch
from torch import nn
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForTokenClassification, 
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)
from evaluate import load as load_metric

# --- 1. KONFIGURACE (Stabilní BERT) ---
model_id = "bert-base-multilingual-cased"
output_dir = "./model_final_pro" 
N_SAMPLES = 50000 

group_map = {
    'NAME': ['FIRSTNAME', 'LASTNAME', 'MIDDLENAME', 'USERNAME', 'ACCOUNTNAME'],
    'LOC': ['CITY', 'STREET', 'ZIPCODE', 'STATE', 'COUNTY', 'BUILDINGNUMBER', 'NEARBYGPSCOORDINATE'],
    'CONTACT': ['EMAIL', 'PHONENUMBER', 'IP', 'IPV4', 'IPV6', 'MAC', 'URL', 'PHONEIMEI'],
    'ID': ['SSN', 'PASSPORT', 'VEHICLEVIN', 'VEHICLEVRM', 'PASSWORD', 'PIN', 'CREDITCARDCVV'],
    'FINANCE': ['CREDITCARDNUMBER', 'IBAN', 'BITCOINADDRESS', 'ETHEREUMADDRESS', 'AMOUNT']
}

# --- 2. WEIGHTED TRAINER (Vynucení vysokého Recallu) ---
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Váhy musí odpovídat dtype výstupu (fix pro Mac MPS)
        weights = torch.ones(
            self.model.config.num_labels, 
            dtype=logits.dtype, 
            device=self.model.device
        )
        
        # PII je 10x důležitější než běžný text (všechny labely kromě indexu 0)
        weights[1:] = 10.0 
        
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        
        # Výpočet loss
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

# --- 3. PŘÍPRAVA DAT ---
print(f"[*] Načítám {N_SAMPLES} vzorců pro stabilní trénink...")
raw_ds = load_dataset("ai4privacy/pii-masking-200k", split='train').select(range(N_SAMPLES))

unique_new_labels = ["O"]
for group in group_map.keys():
    unique_new_labels.extend([f"B-{group}", f"I-{group}"])

label2id = {l: i for i, l in enumerate(unique_new_labels)}
id2label = {i: l for i, l in enumerate(unique_new_labels)}

def map_labels(bio_labels):
    new_labels = []
    for lbl in bio_labels:
        if lbl == "O":
            new_labels.append("O")
            continue
        prefix, clean_name = lbl[:2], lbl[2:]
        for group, members in group_map.items():
            if clean_name in members:
                new_labels.append(prefix + group)
                break
        else: new_labels.append("O")
    return [label2id[l] for l in new_labels]

tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize_and_align(examples):
    tokenized = tokenizer(examples["source_text"], truncation=True, max_length=128, padding='max_length')
    labels = [map_labels(lbls) for lbls in examples["mbert_bio_labels"]]
    tokenized["labels"] = [l[:128] + [-100]*(128-len(l[:128])) for l in labels]
    return tokenized

print("[*] Tokenizace a rozdělení 90/10...")
tokenized_ds = raw_ds.map(tokenize_and_align, batched=True).train_test_split(test_size=0.1)

# --- 4. METRIKY ---
metric = load_metric("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[id2label[p] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    true_labels = [[id2label[l] for (p, l) in zip(pred, lab) if l != -100] for pred, lab in zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"]}

# --- 5. MODEL A TRÉNINK ---
model = AutoModelForTokenClassification.from_pretrained(
    model_id, num_labels=len(unique_new_labels), id2label=id2label, label2id=label2id
)

args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    fp16=False,             # Vypnuto pro maximální stabilitu výpočtů
    max_grad_norm=1.0,      # Zabraňuje "NaN" výbuchům gradientu
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    report_to="none"
)

trainer = WeightedTrainer(
    model=model, 
    args=args, 
    train_dataset=tokenized_ds["train"], 
    eval_dataset=tokenized_ds["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics
)

print("\n🚀 START STABILNÍHO TRÉNINKU (BERT MULTILINGUAL)...")
trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"\n✅ Hotovo! Model je v {output_dir}")"""
import os
import re
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForTokenClassification,
    logging
)

MODEL_NAME = "microsoft/Multilingual-MiniLM-L12-H384"
TAGS = ["O", "B-NAME", "I-NAME", "B-LOC", "I-LOC", "B-CONTACT", "I-CONTACT", "B-ID", "I-ID", "B-FINANCE", "I-FINANCE"]
label2id = {tag: i for i, tag in enumerate(TAGS)}
id2label = {i: tag for i, tag in enumerate(TAGS)}

logging.set_verbosity_info()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def prepare_pii_data(example):
    text = example["source_text"]
    mask = example["privacy_mask"]
    
    tokens = []
    ner_tags = []
    
    words_with_offsets = []
    for m in re.finditer(r'\S+|\n', text):
        words_with_offsets.append((m.group(), m.start(), m.end()))
        
    for word, start, end in words_with_offsets:
        tokens.append(word)
        current_tag = "O"
        
        for pii in mask:
            if start >= pii["start"] and end <= pii["end"]:
                raw_label = pii["label"].upper()
                if "PERSON" in raw_label: label = "NAME"
                elif "LOC" in raw_label or "ADDR" in raw_label: label = "LOC"
                elif "EMAIL" in raw_label or "PHONE" in raw_label: label = "CONTACT"
                elif "BANK" in raw_label or "CARD" in raw_label: label = "FINANCE"
                else: label = "ID"
                
                current_tag = f"B-{label}" if start == pii["start"] else f"I-{label}"
                break
        
        ner_tags.append(label2id.get(current_tag, 0))
        
    return {"tokens": tokens, "ner_tags": ner_tags}

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], 
        truncation=True, 
        is_split_into_words=True, 
        max_length=512
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# --- 4. HLAVNÍ FUNKCE ---
def main():
    print(f"[*] Krok 1: Načítání a konverze dat pro {MODEL_NAME}...")
    raw_dataset = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=False).select(range(50000))    
    processed_dataset = raw_dataset.map(prepare_pii_data, desc="Vytvářím tokens a tags")
    tokenized_dataset = processed_dataset.map(tokenize_and_align_labels, batched=True, desc="Zarovnávám s modelem")

    print("[*] Krok 2: Inicializace modelu...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(TAGS), id2label=id2label, label2id=label2id
    )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"[+] Používám hardware: {device.upper()}")

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=3e-5,
        per_device_train_batch_size=2,      # U MiniLM můžeme zkusit batch 2
        gradient_accumulation_steps=4,      # Efektivní batch 8
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        optim="adamw_torch",
        report_to="none",
        push_to_hub=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    print("\n" + "="*50 + "\n  🚀 START TRÉNINKU (MiniLM Edition)\n" + "="*50)
    
    try:
        trainer.train()
        
        model.save_pretrained("./model_final_pro")
        tokenizer.save_pretrained("./model_final_pro")
        print("\n[+] Model úspěšně uložen do ./model_final_pro")
        
    except Exception as e:
        print(f"\n[!] Chyba při tréninku: {e}")

if __name__ == "__main__":
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    main()