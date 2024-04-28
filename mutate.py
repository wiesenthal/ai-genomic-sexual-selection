import random
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
        model="gpt-3.5-turbo", messages=messages, temperature=0.0, **kwargs
    )
    return completion.choices[0].message.content


def mutate_or_keep_val(val: str | float, mutation_prompt: str, mutation_rate: float):
    if random.random() < mutation_rate:
        print(f"MUTATION HAPPENING: '{val}' is mutating!")
        if isinstance(val, str):
            result = complete(system_prompt=mutation_prompt, user_prompt=val)
            print(f"MUTATION RESULT: '{result}'")
            return result
        elif isinstance(val, float):
            result = complete(system_prompt=mutation_prompt, user_prompt=str(val))
            print(f"MUTATION RESULT: '{result}'")
            # if result is a number, return a float
            if result.isdigit():
                return float(result)
            else:
                print(f"MUTATION FAILED: '{result}' is not a number")
                return val
    else:
        return val
