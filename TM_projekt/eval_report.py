from datasets import load_dataset
from orchestrator import AnonymizationOrchestrator
from tqdm import tqdm

def evaluate_professional():
    """
    Provádí hloubkovou analýzu kvality modelu na základě IR metrik[cite: 2].
    """
    LIMIT = 1000 
    orchestrator = AnonymizationOrchestrator()
    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train", streaming=True).skip(50000)

    tp, fp, fn = 0, 0, 0
    
    for i, example in enumerate(tqdm(dataset, total=LIMIT)):
        if i >= LIMIT: break
        text = example["source_text"]
        true_spans = [(m["start"], m["end"]) for m in example["privacy_mask"]]
        _, pred_findings = orchestrator.run(text)
        pred_spans = [(f["start"], f["end"]) for f in pred_findings]

        for ps in pred_spans:
            if any(max(ps[0], ts[0]) < min(ps[1], ts[1]) for ts in true_spans): tp += 1
            else: fp += 1
        
        for ts in true_spans:
            if not any(max(ps[0], ts[0]) < min(ps[1], ts[1]) for ps in pred_spans): fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 IR REPORT: Precision: {precision:.2%}, Recall: {recall:.2%} (F1: {f1:.2%})")

if __name__ == "__main__":
    evaluate_professional()