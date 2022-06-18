import pandas as pd
import shutil
import requests
import time
import boto3
import os
import sys

styles = sys.argv[1].split(',')

print(styles)

df = pd.read_csv('./artworks.csv')
s3 = boto3.client("s3",aws_access_key_id = 'AKIAROPW2QKXRFZQ6AEG', 
                  aws_secret_access_key = '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') 

# creating a folder for a style on s3
for style in styles:
	print(f'Downloading {style}')
	bucket_name = 'art-styles-dataset'
	object_name = f'raw-data/{style}/'
	create_resp = s3.put_object(Bucket = 'art-styles-dataset', Key = (object_name))

	# creating path in the local
	path = f'/tmp/{style}'
	if not os.path.exists(path):
	    os.makedirs(path)

	images = list(df[df['style'] == style].image)
	if len(images) == 0:
		print('Wrong Entry')

	st = time.time()
	for i, img_url in enumerate(images):
	    
	    # file name
	    filestem = img_url.split('/')[-1].split('.')[0] + '.jpg'    
	    filename = path + filestem
	    
	    # requests for the byte stream
	    img = requests.get(img_url, stream = True)
	    
	    # download the image
	    if img.status_code == 200:
	        img.raw.decode_content = True
	        with open(filename,'wb') as f:
	            shutil.copyfileobj(img.raw, f)
	    else:
	        print('Image Couldn\'t be retreived')
	    
	    # push the image to s3
	    response = s3.upload_file(filename, bucket_name, object_name + filestem)
	    
	    # delete the image
	    if not os.path.exists(filename):
	        os.remove(filename)
	    
	    if i % 100 == 0:
	        end = time.time()
	        print(f'Time for {i}: {end - st}')