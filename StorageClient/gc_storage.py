# Imports the Google Cloud client library
from matplotlib.image import imread
from google.cloud import storage
import numpy as np
import json
import os


# Instantiates a client
client = storage.Client()
bucket_name = 'sample_report_images'
bucket = client.get_bucket(bucket_name)

def list_files(expr=''):	
	'''
		Behaves similarly to ls in bash. Use list_files dir/
		to list all blobs in dir/. Moreover, expr can be used
		as a regexp instead.
	'''
	return [x for x in bucket.list_blobs() if expr in x.name]	

def imload(blob,clean=True):
	'''
		Grabs an img located at the input blob.
		If clean=True, then does not save image
		locally. If false, then saved to current dir.
	'''
	fname = blob.name.split('/')[-1]
	with open(fname, "wb") as fw:
		blob.download_to_file(fw)	
	image = imread(fname)
	if clean:
		os.remove(fname)
	return image

def get_file(pattern,files=[]):
	if files == []:
		files = list_files(pattern)
	else:
		files = [x for x in files if pattern in x.name]
	return files

def get_img(report,files=[],img_dir='diagrams_with_google_label/',clean=True):
	'''
		Gets an image by report ID
	'''
	if img_dir[-1]!='/':
		img_dir = img_dir+'/'
	if type(report) == str:
		files = get_file(img_dir+report,files)
		if len(files) == 0:		
			return np.NaN
		return imload([x for x in files if '.png' in x.name][0],clean)
	elif type(report) == list:
		files = [get_file(img_dir+x,files) for x in report]
		if all(len(x)==0 for x in files):
			return np.NaN
		return [imload([x for x in file if '.png' in x.name][0],clean) for file in files]
	else:
		return np.NaN
	
def get_report(report,files=[],img_dir='diagrams_with_google_label/',clean=True):
	'''
		Gets a report by report ID
	'''
	if (img_dir[-1]!='/'):
		img_dir = img_dir+'/'
	if type(report) == str:
		files = get_file(img_dir+report,files)
		if len(files) == 0:		
			return np.NaN
		return imload([x for x in files if '.jpeg' in x.name][0],clean)
	elif type(report) == list:
		files = [get_file(img_dir+x,files) for x in report]
		if all(len(x)==0 for x in files):
			return np.NaN
		return [imload([x for x in file if '.jpeg' in x.name][0],clean) for file in files]
	else:
		return np.NaN

def get_negatives(report,files=[],neg_dir='negatives/'):
	'''
		Gets a list of the negatives for a given report
	'''
	if neg_dir[-1]!='/':
		neg_dir = neg_dir+'/'
	files = get_file(neg_dir+report,files)
	return json.loads(files[0].download_as_string())[report]
