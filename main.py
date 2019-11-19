import argparse
import os
import base64
import io
import json

from PIL import Image
import requests

IMAGE_EXTENSIONS = ["bmp", "jpg", "jpeg", "png"]
URL = "http://localhost:{}/v1/models/default:predict"
PORT = 8501

def get_image_paths(folder):
    return [os.path.join(source,x) for x in os.listdir(folder) if x.split(".")[-1].lower() in IMAGE_EXTENSIONS]

def get_base64_string(image_path):
    image = Image.open(image_path)
    with io.BytesIO() as buffer:
        image.save(buffer, 'png', quality=100)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    # # Google's code
    # with io.open(image_path, 'rb') as image_file:
    #     encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    # return encoded_image

def get_prediction(data):
    response = requests.post(URL.format(PORT), data=data)
    return response

def construct_request(img_path, img_str):
    request = '{"image_bytes":{"b64": "'+str(img_str)+'"},"key": "'+img_path+'"}'
    return request


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process images to get predictions from AutoML Edge Containerized Models.')
    parser.add_argument('directory', help='directory which stores the images')
    parser.add_argument('batch_size', type=int, help='size of the prediction batch')

    args = parser.parse_args()
    source = args.directory
    batch_size = args.batch_size

    images = get_image_paths(source)

    records = []
    for i in range(0, len(images), batch_size):
        image_requests = []
        for j in range(batch_size):
            img_str = get_base64_string(images[i+j])
            image_requests.append(construct_request(images[i+j], img_str))

        wrapped_request = '{"instances": '+ "[{}]".format(", ".join([str(request) for request in image_requests])) +'}'
        # print(wrapped_request)

        response = get_prediction(data=wrapped_request)
        response_json = json.loads(response.content)
        print(response_json)
        records.append(response_json)
    records = {"records": records}
    with open("records.json", "w") as f:
        f.write(json.dumps(records))