from openai import OpenAI
client = OpenAI()

def complete(system_prompt: str, user_prompt: str):
  messages = []
  if system_prompt:
    messages.append({"role": "system", "content": system_prompt})
  if user_prompt:
    messages.append({"role": "user", "content": user_prompt})
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages
  )
  return completion.choices[0].message.content
