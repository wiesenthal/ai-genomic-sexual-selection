from flask import Flask
import os
import json

app = Flask(__name__)

@app.route('/')
def display_images():
    image_data_dir = os.path.join(os.getcwd(), 'image_data')
    image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.json')]
    # sort by filename
    image_files.sort()
    images_content = {}
    for image_file in image_files:
        with open(os.path.join(image_data_dir, image_file), 'r') as file:
            images_content[image_file] = json.load(file)
    html_content = "<html><body>"
    for image_file, content in images_content.items():
        html_content += f"<h3>{image_file}</h3>"
        # find keys
        keys = content.keys()
        # get first key
        key = next(iter(keys))
        # get value
        data = content[key]
        for key, value in data.items():
            html_content += f"<p>{key}: <img src={value}></p>"
    html_content += "</body></html>"
    return html_content

if __name__ == '__main__':
    app.run(debug=True)
