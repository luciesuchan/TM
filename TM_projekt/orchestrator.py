"""import os
from agents.pattern_agent import PatternAgent
from agents.context_agent import ContextAgent
from agents.gliner_agent import GLiNERAgent

class AnonymizationOrchestrator:
    def __init__(self, model_path="./model_final_pro"):
        self.agents = [PatternAgent(), ContextAgent(model_path), GLiNERAgent()]

    def run(self, text: str):
        if not text or not text.strip(): return text, []

        all_findings = []
        for agent in self.agents:
            all_findings.extend(agent.find_pii(text))

        if not all_findings: return text, []

        # Seřazení podle pozice
        all_findings.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        # --- SLUČOVÁNÍ A HLASOVÁNÍ ---
        merged = []
        if all_findings:
            curr = dict(all_findings[0])
            # Sledujeme, kolik agentů danou věc našlo
            sources = {curr["source"]}
            
            for nxt in all_findings[1:]:
                gap_text = text[curr["end"]:nxt["start"]]
                has_sentence_end = any(char in gap_text for char in [".", "!", "?"])

                if nxt["start"] <= curr["end"] + 4 and not has_sentence_end:
                    curr["end"] = max(curr["end"], nxt["end"])
                    sources.add(nxt["source"])
                    
                    # Priorita labelů
                    if nxt["source"] == "PatternAgent":
                        curr["label"] = nxt["label"]
                    elif nxt["source"] == "GLiNERAgent" and curr["source"] == "ContextAgent":
                        curr["label"] = nxt["label"]
                else:
                    curr["hit_count"] = len(sources)
                    merged.append(curr)
                    curr = dict(nxt)
                    sources = {curr["source"]}
            
            curr["hit_count"] = len(sources)
            merged.append(curr)

        # --- 🛡️ DRACONIAN PRECISION FILTER ---
        final_merged = []
        for f in merged:
            ent_text = text[f["start"]:f["end"]].strip()
            
            # 1. Regexy (PatternAgent) bereme vždy - jsou neomylné
            if "PatternAgent" in f.get("source", ""):
                final_merged.append(f)
                continue
            
            # 2. Pokud se shodnou dva a více agentů, je to téměř jistě PII (Precision boost)
            if f.get("hit_count", 1) >= 2:
                final_merged.append(f)
                continue
                
            # 3. Pokud je to jen jeden model, musí splnit přísná kritéria
            # Vyhodíme krátká slova (šum) a věci bez velkých písmen/číslic
            if len(ent_text) > 3 and (any(c.isupper() for c in ent_text) or any(c.isdigit() for c in ent_text)):
                final_merged.append(f)

        # Aplikace maskování odzadu
        processed_text = text
        for f in sorted(final_merged, key=lambda x: x["start"], reverse=True):
            processed_text = processed_text[:f["start"]] + f"<{f['label']}>" + processed_text[f["end"]:]

        return processed_text, final_merged"""

import os
from agents.pattern_agent import PatternAgent
from agents.context_agent import ContextAgent
from agents.gliner_agent import GLiNERAgent

class AnonymizationOrchestrator:
    def __init__(self, model_path="./model_final_pro"):
        self.agents = [PatternAgent(), ContextAgent(model_path), GLiNERAgent()]

    def run(self, text: str):
        if not text or not text.strip(): return text, []

        all_findings = []
        for agent in self.agents:
            all_findings.extend(agent.find_pii(text))

        if not all_findings: return text, []

        all_findings.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))

        merged = []
        curr = dict(all_findings[0])
        sources = {curr["source"]}
        
        for nxt in all_findings[1:]:
            gap_text = text[curr["end"]:nxt["start"]]
            has_sentence_end = any(char in gap_text for char in [".", "!", "?"])

            if nxt["start"] <= curr["end"] + 4 and not has_sentence_end:
                sources.add(nxt["source"])
                curr["end"] = max(curr["end"], nxt["end"])
                
                if nxt["source"] == "PatternAgent":
                    curr["label"] = nxt["label"]
                elif nxt["source"] == "GLiNERAgent" and curr["source"] == "ContextAgent":
                    curr["label"] = nxt["label"]
            else:
                curr["hit_count"] = len(sources)
                merged.append(curr)
                curr = dict(nxt)
                sources = {curr["source"]}
        
        curr["hit_count"] = len(sources)
        merged.append(curr)
        
        # --- FINÁLNÍ TUNING (Precision 95+) ---
        final_merged = []
        for f in merged:
            ent_text = text[f["start"]:f["end"]].strip()
            
            # 1. Regexy (PatternAgent) bereme hned
            if f["source"] == "PatternAgent":
                final_merged.append(f)
                continue
            
            # 2. Shoda více agentů (Consensus) = 100% jistota
            if f.get("hit_count", 1) >= 2:
                final_merged.append(f)
                continue

            # 3. Jeden agent (musí být aspoň 3 znaky a mít velké písmeno nebo číslo)
            is_strong = (len(ent_text) >= 3 and ent_text[0].isupper()) or any(c.isdigit() for c in ent_text)
            
            if is_strong:
                final_merged.append(f)

        processed_text = text
        for f in sorted(final_merged, key=lambda x: x["start"], reverse=True):
            processed_text = processed_text[:f["start"]] + f"<{f['label']}>" + processed_text[f["end"]:]

        return processed_text, final_merged