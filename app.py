from PIL import Image
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
import image_client
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage import img_as_float, img_as_ubyte
from skimage import io
import boto3
import numpy as np
from PIL import Image
import shutil
import os
import json
from flask import Flask
from flask_cors import CORS, cross_origin

def find_bucket_key(s3_path, id):

    s3_components = s3_path.split('/')
    bucket = s3_components[2].split('.')
    bucket_name = bucket[0]
    key = s3_components[len(s3_components)-1]
    conn = boto3.client('s3','ap-southeast-1')
    response = conn.get_object(Bucket=bucket_name, Key=key)
    body = response['Body']
    image1 = Image.open(body)
    image = image1.resize((224, 224))
    image =np.array(image1)    
    image = img_as_float(image).astype(np.float32)
    sigma_est = np.mean(estimate_sigma(image, multichannel=True))
    patch_kw = dict(patch_size=10,      # 5x5 patches
                            patch_distance=3,  # 13x13 search area
                            multichannel=True)

    denoise_img = denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=False,
                                    **patch_kw)
    

    io.imsave("preprocessedimage_"+str(id)+".jpg", denoise_img)  
    return image


app = Flask(__name__)
CORS(app, resources={r"/home": {"origins": "*"}})


@app.route('/home', methods=['POST'])
#@cross_origin()
def home():

    if request.method == 'POST':

        data = request.get_json()

        outputlist =[]

        for i in range (0,len(data)):

            id= data[i].get('id')
            s3_bucketpath=data[i].get('path')
            find_bucket_key (s3_bucketpath, id)
            inference_model = data[i].get('model')  

            filepath ="preprocessedimage_"+str(id)+".jpg"  
                
            triton_output, confidence= image_client.triton_inference(verbose= False,
                    async_set= False,
                    streaming= False,
                    model_name= inference_model,
                    model_version = "",
                    batch_size= 1,
                    classes= 1,
                    scaling= 'INCEPTION',
                    url= 'Trito-Servi-2XD8EHC3N2RV-1640060134.ap-southeast-1.elb.amazonaws.com',
                    protocol= 'HTTP',
                    image_filename= filepath )
            
            dict_out = {'id':id,
                    'inference_model' : inference_model,
                    'triton_output' : triton_output,
                    'confidence' : confidence,
                    's3_path': s3_bucketpath
                    }

            outputlist.append(dict_out)

        return json.dumps(outputlist)


if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0',port=5000)