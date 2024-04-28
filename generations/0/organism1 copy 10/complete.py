from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def complete(system_prompt: str, user_prompt: str, **kwargs):
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  if user_prompt:
    messages.append({"role": "user", "content": user_prompt})
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    temperature=0.0,
    **kwargs
  )
  return completion.choices[0].message.content
