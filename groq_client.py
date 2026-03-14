import os
import logging
from typing import List, Optional, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from groq import Groq

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIX: Removed the incorrect string inside load_dotenv()
load_dotenv() 
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing! Ensure your file is named exactly '.env' and contains GROQ_API_KEY=your_key")

client = Groq(api_key=API_KEY)
logger.info(f"✅ Groq connection established")

class RobustGroqClient:
    def __init__(self):
        self.available_models = self._discover_models()
        self.stats = {"requests": 0, "models_tried": 0, "fallbacks": 0}
    
    def _discover_models(self) -> List[str]:
        try:
            models_response = client.models.list()
            chat_models = [
                m.id for m in models_response.data 
                if 'chat' in m.id.lower() or 'llama' in m.id.lower() or 'gemma' in m.id.lower()
            ][:5]
            return chat_models if chat_models else ["llama3-8b-8192"]
        except:
            return ["llama3-8b-8192", "mixtral-8x7b-32768"]
    
    @lru_cache(maxsize=50)
    def _try_model(self, prompt: str, model: str, max_tokens: int) -> Optional[str]:
        try:
            chat = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens,
                timeout=15
            )
            self.stats["models_tried"] += 1
            self.stats["requests"] += 1
            return chat.choices[0].message.content.strip()
        except:
            return None
    
    def generate_question(self, skill: str, role: str) -> Tuple[str, str]:
        prompt = f"Create 1 interview question for {role} about {skill}. Question only:"
        for model in self.available_models:
            result = self._try_model(prompt, model, 80)
            if result: return result, model
        return f"How do you use {skill}?", "fallback"
    
    def score_answer(self, answer: str, skill: str) -> float:
        prompt = f"Score this answer about {skill} from 0.0 to 1.0 (number only): {answer[:300]}"
        for model in ["llama3-8b-8192"] + self.available_models:
            result = self._try_model(prompt, model, 8)
            if result:
                try: return min(1.0, max(0.0, float(result.strip())))
                except: pass
        return 0.5

groq_client = RobustGroqClient()
