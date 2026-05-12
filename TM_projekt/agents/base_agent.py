import re
from transformers import pipeline

class BaseAgent:
    """Základní třída pro všechny agenty"""
    def process(self, text, metadata):
        pass