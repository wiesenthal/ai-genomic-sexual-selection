from flask import Flask, jsonify, render_template, make_response
import json
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# DO NOT COMMIT
client = OpenAI()


app = Flask(__name__)

client = OpenAI()


def load_descriptions(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return {name: person["phenotype"] for name, person in data.items()}


import time


def generate_image(description):
    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=description,
            size="512x512",
            quality="standard",
            n=1,
        )
    except Exception as e:
        return ""
    return response.data[0].url


def generate_images_from_generation(data, generation_number):
    print(f"Generating images for generation {generation_number}")
    descriptions = {name: person["phenotype"] for name, person in data.items()}
    with ThreadPoolExecutor() as executor:
        future_to_name = {
            executor.submit(generate_image, description): name
            for name, description in descriptions.items()
        }
        image_urls = {
            f"{future_to_name[future]}": future.result() for future in future_to_name
        }
    generations = {f"{generation_number}": image_urls}
    # write to file
    with open(
        f"/Users/miles/Documents/github/ai-genomic-sexual-selection/image_data/images-generation-{generation_number}.json",
        "w",
    ) as file:
        json.dump(generations, file)


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/')
# def index():
#     response = make_response("Hello World")
#     return response

# images = {}

# @app.route('/generate-images', methods=['GET'])
# def generate_images():
#     descriptions = load_descriptions('./generations/expressions.json')  # Adjust the path as needed
#     image_urls = {name: generate_image(description) for name, description in descriptions.items()}
#     images.update(image_urls)
#     return jsonify(image_urls)

# # if __name__ == '__main__':
# #     app.run(debug=True)

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
