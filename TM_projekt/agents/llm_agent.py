import google.generativeai as genai
import json
import re

class LLMAgent:
    def __init__(self, api_key="AIzaSyC09lmcrSggLDyGGsC2zObczJrWjxppVk8"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def find_pii(self, text: str):
        prompt = f"""
        Jsi expert na ochranu osobních údajů (GDPR). Tvým úkolem je v textu identifikovat PII (osobní údaje).
        Vrať výsledek POUZE jako JSON seznam objektů: {{"start": int, "end": int, "label": str, "text": str}}.
        Hledej: NAME, LOC, CONTACT, ID, FINANCE.
        
        TEXT: "{text}"
        """
        
        try:
            response = self.model.generate_content(prompt)
            clean_json = re.search(r'\[.*\]', response.text, re.DOTALL).group()
            findings = json.loads(clean_json)
            for f in findings:
                f["source"] = "LLMAgent"
            return findings
        except Exception as e:
            print(f"LLM Error: {e}")
            return []