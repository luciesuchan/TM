"""
Demonstrace anonymizátoru na online datasetu ai4privacy/pii-masking-200k.

Spuštění:
  python main.py

Pokud model ještě není natrénován, PatternAgent funguje samostatně
a ContextAgent je tiše přeskočen.
"""
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import AnonymizationOrchestrator
from datasets import load_dataset


def main():
    print("=" * 60)
    print("🔍 AGENTNÍ ANONYMIZÁTOR")
    print("=" * 60)

    MODEL_PATH = "./moje_pii_model"
    N_SAMPLES  = 5

    # Inicializace systému
    print("[*] Inicializuji agenty...")
    orchestrator = AnonymizationOrchestrator(model_path=MODEL_PATH)
    print("✅ Systém připraven.\n")

    # Načtení datasetu
    print("[*] Načítám dataset (streaming)...")
    dataset = load_dataset(
        "ai4privacy/pii-masking-200k",
        split="train",
        streaming=True,
    )

    print(f"[*] Zpracovávám {N_SAMPLES} vzorků...\n")
    count = 0

    for example in dataset:
        if count >= N_SAMPLES:
            break

        # Vytáhneme zdrojový text
        raw_text = example.get("source_text", "")
        if not raw_text:
            continue

        # Anonymizace
        anonymized, findings = orchestrator.run(raw_text)

        # Výpis
        print(f"📍 VZOREK #{count + 1}")
        print(f"   VSTUP : {raw_text[:120].strip()}")
        print(f"   VÝSTUP: {anonymized[:120].strip()}")

        pattern_hits = sum(1 for f in findings if f["source"] == "PatternAgent")
        context_hits = sum(1 for f in findings if f["source"] == "ContextAgent")
        labels_found = [f["label"] for f in findings]
        print(f"   Nálezy: PatternAgent={pattern_hits}x, ContextAgent={context_hits}x")
        if labels_found:
            print(f"   Labely: {labels_found}")
        print("-" * 50)

        count += 1

    print(f"\n✅ Zpracováno {count} vzorků.")


if __name__ == "__main__":
    main()"""

import os
from datasets import load_dataset
from orchestrator import AnonymizationOrchestrator
from tqdm import tqdm

def evaluate_professional():
    # 1. NASTAVENÍ - tady se to zastaví
    LIMIT = 100 
    MODEL_PATH = "./model_final_pro"
    orchestrator = AnonymizationOrchestrator(model_path=MODEL_PATH)
    
    # Načteme testovací data (přeskočíme prvních 2000, na kterých se model učil)
    print("[*] Načítám testovací data...")
    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True).skip(2000)

    tp, fp, fn = 0, 0, 0
    processed_count = 0

    print(f"[*] Spouštím evaluaci na {LIMIT} vzorcích...")

    # Použijeme tqdm pro hezký progress bar
    for example in tqdm(dataset, total=LIMIT):
        if processed_count >= LIMIT:
            break
        
        text = example["source_text"]
        # Ground Truth (Pravda z datasetu)
        true_spans = [(m["start"], m["end"]) for m in example["privacy_mask"]]
        
        # Predikce (Co našel tvůj systém)
        _, findings = orchestrator.run(text)
        pred_spans = [(f["start"], f["end"]) for f in findings]

        # Výpočet metrik (Přesná shoda pozic)
        for ps in pred_spans:
            if ps in true_spans:
                tp += 1
            else:
                fp += 1
        
        for ts in true_spans:
            if ts not in pred_spans:
                fn += 1
            
        processed_count += 1

    # VÝPOČET METRIK
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "="*40)
    print("📊 FINÁLNÍ TEXT-MINING REPORT")
    print("="*40)
    print(f"Zpracováno vět:   {processed_count}")
    print("-" * 40)
    print(f"Precision (Přesnost): {precision:.2%}")
    print(f"Recall (Úplnost):    {recall:.2%}")
    print(f"F1-Score:           {f1:.2%}")
    print("="*40)
    print("\nPROFI TIP: Tyto hodnoty vlož do závěru dokumentace jako důkaz úspěšnosti fine-tuningu.")

if __name__ == "__main__":
    evaluate_professional()