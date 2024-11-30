import deepl
from dotenv import load_dotenv
import os

load_dotenv()

class Translator:
    def __init__(self):
        self.translator = deepl.Translator(os.getenv("DEEPL_API_KEY"))
    
    def translate_text(self, text, target_language):
        if text:
            translation = self.translator.translate_text(text, target_lang=target_language)
            return translation.text
        else:
            return text
    
    def translate_JSON(self, data, target_language):
        if isinstance(data, dict):
            translated_data = {}
            for key, value in data.items():
                translated_data[key] = self.translate_JSON(value, target_language)
            return translated_data
        elif isinstance(data, list):
            translated_data = []
            for item in data:
                translated_data.append(self.translate_JSON(item, target_language))
            return translated_data
        elif isinstance(data, str):
            return self.translate_text(data, target_language)
        else:
            return data

