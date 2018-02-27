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

# Get image and report from a given ID
report_id = '9021-2015-0001'
img = get_img(report_id,pos_files,img_dir)
report = get_report(report_id,pos_files,img_dir)

# Get negatives (won't work until all files are loaded)
negs = get_negatives(report_id,neg_files,neg_dir)
neg_img = get_img(negs,pos_files,img_dir)
neg_report = get_report(negs,pos_files,img_dir)

# If you want to actually save the image locally as well,
# Then run get_image with another paramater as 
img = get_img(report_id,pos_files,img_dir,False)
report = get_report(report_id,pos_files,img_dir,False)
