from google import genai
import time
import sys
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

def generate_text(prompt):
    response = client.models.generate_content_stream(model="gemini-2.5-flash",contents=prompt)
    for chunk in response:
        if chunk.text:
            for char in chunk.text:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(0.005)
                
generate_text("write 5 lines of paragraph blog on AI Agnet")