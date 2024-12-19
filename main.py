import json
import torch
import openai
import gc
import os
import logging
import time
import http.client
from typing import List, Dict
from dataclasses import dataclass
from retry import retry
import yaml
from PIL import Image
import base64
from io import BytesIO
import requests

from platform_backends.xhs_backends import sign, test_create_simple_note
from xhs import XhsClient

# Configuration handling
@dataclass
class Config:
    api_keys: Dict
    news_url: str
    output_dir: str
    cache_dir: str
    LLM_choose: str
    T2I_choose: str
    styles: Dict
    platform_settings: Dict

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

    @retry(tries=3, delay=2, logger=logger)
    def get_daily_news(self) -> List[str]:
        logger.info("Fetching daily news...")
        response = requests.get(self.config.news_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data['data']['news']

    def translate_to_prompt(self, news_text: str) -> Dict:
        logger.info("Generating prompts from news...")
        client = openai.OpenAI(
            api_key=self.config.api_keys['LLM'][self.config.LLM_choose]['api_key'],
            base_url=self.config.api_keys['LLM'][self.config.LLM_choose]['base_url']
        )

        system_prompt = """
        You are a professional Stable Diffusion prompt engineer specializing in:
        1. Analyzing news content for visual elements
        2. Creating detailed artistic prompts
        3. Incorporating diverse artistic styles
        4. Maintaining emotional impact
        5. Please note that the news is taken from the China, if not specifically emphasize the characters from other countries, please indicate in the prompt that Chinese elements are drawn, such as Chinese teachers, Chinese elderly, etc
        6. Avoid political symbols or figures in prompts
        7. Prompting in standard and easy-to-understand English
        
        Output json Format: {"prompts": [{"text": prompt, "style": style_name}]}
        """

        response = client.chat.completions.create(
            model=self.config.api_keys['LLM'][self.config.LLM_choose]['model'],
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": news_text}],
            temperature=0.7,
            max_tokens=4095,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def translate_to_comment(self, news_text: str) -> Dict:
        logger.info("Generating comments from news...")
        client = openai.OpenAI(
            api_key=self.config.api_keys['LLM'][self.config.LLM_choose]['api_key'],
            base_url=self.config.api_keys['LLM'][self.config.LLM_choose]['base_url']
        )

        system_prompt = """
        你需要扮演一名专业的中文新闻撰稿人，介绍新闻中涉及的专业概念，注意只在概念比较晦涩时才介绍；最后，你需要使用中文详细评价每一条新闻报道的信息对社会生活、经济、民生的影响，特别注意越详细越好。你需要尽量像你扮演的角色，因此你需要保持文字的多样性，特别注意要避免出现相同句式。每一条新闻的评述写在各自对应的“comment”中。
        最后你需要简短总结其中最有吸引力的新闻，填写在title中。title应该控制在10字以内。

        Output json Format: {"title": title, "comments":[{"text": comment}, {"text": comment}, ...]}
        """

        response = client.chat.completions.create(
            model=self.config.api_keys['LLM'][self.config.LLM_choose]['model'],
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": news_text}],
            temperature=0.7,
            max_tokens=4095,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)

    def generate_images(self, prompts: Dict, model_name: str):
        logger.info(f"Generating images using model: {model_name}...")
        metadata = []
        image_paths = []
        model_config = self.config.api_keys['Text2Image'][model_name]
        if not model_config:
            raise ValueError(f"Model configuration for '{model_name}' not found.")

        for idx, prompt_data in enumerate(prompts['prompts']):
            try:
                if model_config['type'] == 'Local':
                    # Call local model API using http.client
                    conn = http.client.HTTPConnection(model_config['api_url'], int(model_config['port']))
                    payload = json.dumps({
                        "prompt": prompt_data['text'] + f", {self.config.styles.get(prompt_data['style'], {})}",
                        "num_inference_steps": 40,
                        "guidance_scale": 4.5,
                        "style": {}
                    })
                    headers = {"Content-Type": "application/json"}
                    conn.request("POST", "/generate", body=payload, headers=headers)
                    response = conn.getresponse()
                    data = response.read().decode()
                    conn.close()
                elif model_config['type'] == 'remote':
                    # Call remote API using http.client
                    conn = http.client.HTTPConnection(model_config['url'].split('/')[2])
                    payload = json.dumps({
                        "prompt": prompt_data['text'],
                        "num_inference_steps": 40,
                        "guidance_scale": 4.5,
                        "style": self.config.styles.get(prompt_data['style'], {})
                    })
                    headers = {"Authorization": f"Bearer {model_config['api_key']}", "Content-Type": "application/json"}
                    conn.request("POST", model_config['url'], body=payload, headers=headers)
                    response = conn.getresponse()
                    data = response.read().decode()
                    conn.close()

                response_data = json.loads(data)
                image_base64 = response_data['image_base64']
                image = Image.open(BytesIO(base64.b64decode(image_base64)))

                # Save with metadata
                timestamp = int(time.time())
                filename = f"news_art_{timestamp}_{idx}.png"
                image.save(os.path.join(self.config.output_dir, filename))
                image_paths.append(os.path.join(self.config.output_dir, filename))

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

        return image_paths

    def push_to_xhs(self, image_paths, news_captions, news_title):
        logger.info(f"Pushing news to xhs...")
        try:
            cookie = self.config.platform_settings['xhs']['cookie']
            xhs_client = XhsClient(cookie, sign=sign)
            response = test_create_simple_note(xhs_client, image_paths=image_paths, desc=news_captions, title=news_title)
            return response
        except Exception as e:
            logger.error(f"Error pushing to xhs: {str(e)}")

    def run(self):
        try:
            news = self.get_daily_news()
            prompts = self.translate_to_prompt(','.join(news))
            comments = self.translate_to_comment(','.join(news))
            # print(comments)
            # print('每日AI新闻速递：'+','.join(news))
            # print(dict(comments).get('title','今日新闻'))
            # print(comments)
            image_paths = self.generate_images(prompts, self.config.T2I_choose)
            self.push_to_xhs(image_paths, '每日AI新闻速递：'+'\n'.join(news), '60s读懂世界：' + dict(comments).get('title','今日新闻'))

            logger.info("Generation completed successfully")
        except Exception as e:
            logger.error(f"Error in generation pipeline: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='News Art Generator')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    args = parser.parse_args()

    generator = NewsArtGenerator(args.config)

    import schedule
    import time
    schedule.every().day.at("07:00").do(generator.run)
    while True:
        schedule.run_pending()
        time.sleep(1)
    #generator.run()
