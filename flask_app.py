from flask import Flask
import os
import json
import re

app = Flask(__name__)

@app.route('/')
def display_images():
    image_data_dir = os.path.join(os.getcwd(), 'image_data')
    image_files = [f for f in os.listdir(image_data_dir) if f.endswith('.json')]
    # Sort by filename
    image_files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    images_content = {}
    for image_file in image_files:
        with open(os.path.join(image_data_dir, image_file), 'r') as file:
            images_content[image_file] = json.load(file)

    # Start HTML content with a header and include CSS for the grid
    html_content = '''
    <html>
    <head>
        <style>
            .grid-container {
                display: grid;
                grid-template-columns: repeat(4, 1fr); /* Display 2 images per row */
                gap: 10px; /* Add some space between images */
                margin-bottom: 20px; /* Space below each set of images */
            }
            .grid-item {
                text-align: center; /* Center-align the image title and image */
                margin: 5px; /* Adds some space around each grid item */
            }
            .grid-item img {
                width: 100%; /* Make images expand to fill the grid item */
                height: auto; /* Keep the aspect ratio */
                display: block; /* Ensure that images are block-level elements */
                margin: 0 auto; /* Center the image in the grid item */
            }
            .section-title {
                text-align: center; /* Center-align the section title */
                margin-top: 20px; /* Space above the section title */
            }
        </style>
    </head>
    <body>
    '''
    # Loop through each image file and group images under their titles
    for image_file, content in images_content.items():
        html_content += f"<div class='section-title'><h2>{image_file.replace('.json', '')}</h2></div><div class='grid-container'>"
        keys = content.keys()
        for key in keys:
            data = content[key]
            for name, url in data.items():
                html_content += f"<div class='grid-item'><p>{name}</p><img src='{url}'></div>"
        html_content += "</div>"  # Close the grid-container div

    html_content += "</body></html>"
    return html_content

if __name__ == '__main__':
    app.run(debug=True)
