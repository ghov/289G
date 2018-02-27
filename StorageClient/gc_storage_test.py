#########################################################################
# Seriously, don't just run this with python3 gc_storage_test.py 		#
# Open python and run each block at a time to see what it does.			#
# Each block is a different way you can pull data from datastore.		#
# Note, that once they load all the data I will move their data around	#
# to make this run a bit faster, but until then; get over it.     		#			
#########################################################################

from gc_storage import list_files,get_img,get_report,get_negatives

# select one of their many negatives/ directories
neg_dir = 'negatives_25000_no_constraint' 
img_dir = 'diagrams_with_google_label'

# Get the list of all blobs (files)
files = list_files()
pos_files = [x for x in files if img_dir in x.name]
neg_files = [x for x in files if neg_dir in x.name]
# Note you could also do, but this is about twice as long
# pos_files = list_files('diagrams_with_google_label/')
# neg_files = list_files(neg_dir)

# Note that these are blob classes. To see the path
# of each blob, look at its name variable with
pos_files[1].name

# Get image and report from a given ID
report_id = '9021-2015-0001'
img = get_img(report_id,pos_files,img_dir)
report = get_report(report_id,pos_files,img_dir)

# Get negatives for that ID (won't work until all files are loaded)
negs = get_negatives(report_id,neg_files,neg_dir)
neg_img = get_img(negs,pos_files,img_dir)
neg_report = get_report(negs,pos_files,img_dir)

# If you want to actually save the image locally as well,
# Then run get_image with another parameter as 
img = get_img(report_id,pos_files,img_dir,False)
report = get_report(report_id,pos_files,img_dir,False)

# If you want to save all files with 9226 in the report id
# then you can use the imload(x,False) function as 
from gc_storage import imload
_ = [imload(x,False) for x in pos_files if '/9160-' in x.name]
