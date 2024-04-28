from flask import Flask, jsonify, render_template, make_response
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


app = Flask(__name__)

client = OpenAI()

def load_descriptions(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return {name: person['phenotype'] for name, person in data.items()}

def generate_image(description):
    response = client.images.generate(
        model="dall-e-3",
        prompt=description,
        size="1024x1024",
        quality="standard",
        n=1
    )
    return response.data[0].url

# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/')
def index():
    response = make_response("Hello World")
    return response

@app.route('/generate-images', methods=['GET'])
def generate_images():
    descriptions = load_descriptions('../generations/expressions.json')  # Adjust the path as needed
    image_urls = {name: generate_image(description) for name, description in descriptions.items()}
    return jsonify(image_urls)

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
