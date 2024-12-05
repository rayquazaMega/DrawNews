import json
import torch
import requests
import openai
import gc
import os
import logging
import time
from typing import List, Dict
from dataclasses import dataclass
from retry import retry
import yaml
from PIL import Image
from PIL import Image
import io
import base64


# Configuration handling
@dataclass
class Config:
    api_key: str
    base_url: str
    model: str
    news_url: str
    output_dir: str
    cache_dir: str
    styles: Dict
    models: Dict  # Add a new field for different models

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsArtGenerator:
    def __init__(self, config_path: str):
        self.config = Config.from_yaml(config_path)
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.cache_dir, exist_ok=True)

    @retry(max_tries=3, delay=2, logger=logger)
    def get_daily_news(self) -> List[str]:
        logger.info("Fetching daily news...")
        response = requests.get(self.config.news_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data']['news']

    def translate_to_prompt(self, news_text: str) -> Dict:
        logger.info("Generating prompts from news...")
        client = openai.OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )

        system_prompt = """
        You are a professional Stable Diffusion prompt engineer specializing in:
        1. Analyzing news content for visual elements
        2. Creating detailed artistic prompts
        3. Incorporating diverse artistic styles
        4. Maintaining emotional impact

        Output json Format: {"prompts": [{"text": prompt, "style": style_name}]}
        """

        response = client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": news_text}
            ],
            temperature=0.7,
            max_tokens=4095,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def generate_images(self, prompts: Dict, model_name: str):
        logger.info(f"Generating images using model: {model_name}...")
        metadata = []
        model_config = self.config.models.get(model_name)
        if not model_config:
            raise ValueError(f"Model configuration for '{model_name}' not found.")

        for idx, prompt_data in enumerate(prompts['prompts']):
            try:
                if model_config['type'] == 'local':
                    # Call local model API
                    response = requests.post(
                        model_config['url'] + "/generate",
                        json={
                            "prompt": prompt_data['text'],
                            "num_inference_steps": 40,
                            "guidance_scale": 4.5,
                            "style": self.config.styles.get(prompt_data['style'], {})
                        },
                        timeout=60
                    )
                elif model_config['type'] == 'remote':
                    # Call remote API
                    response = requests.post(
                        model_config['url'],
                        headers={"Authorization": f"Bearer {model_config['api_key']}"},
                        json={
                            "prompt": prompt_data['text'],
                            "num_inference_steps": 40,
                            "guidance_scale": 4.5,
                            "style": self.config.styles.get(prompt_data['style'], {})
                        },
                        timeout=60
                    )
                else:
                    raise ValueError(f"Unsupported model type: {model_config['type']}")

                response.raise_for_status()
                image_base64 = response.json()['image_base64']
                image = Image.open(io.BytesIO(base64.b64decode(image_base64)))

                # Save with metadata
                timestamp = int(time.time())
                filename = f"news_art_{timestamp}_{idx}.png"
                image.save(os.path.join(self.config.output_dir, filename))

                metadata.append({
                    'filename': filename,
                    'prompt': prompt_data['text'],
                    'style': prompt_data['style'],
                    'timestamp': timestamp
                })

                # Memory management
                gc.collect()

            except Exception as e:
                logger.error(f"Error generating image {idx}: {str(e)}")
                continue

        # Save metadata
        with open(os.path.join(self.config.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def run(self):
        try:
            news = self.get_daily_news()
            prompts = self.translate_to_prompt(','.join(news))
            self.generate_images(prompts, self.model_name)
            logger.info("Generation completed successfully")
        except Exception as e:
            logger.error(f"Error in generation pipeline: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='News Art Generator')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    generator = NewsArtGenerator(args.config)
    generator.run()
