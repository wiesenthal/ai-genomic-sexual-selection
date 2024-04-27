from openai import OpenAI
client = OpenAI()

def complete(prompt: str):
  completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
      {"role": "user", "content": prompt}
    ]
  )
  return completion.choices[0].message.content

print(complete("Hello!"))

