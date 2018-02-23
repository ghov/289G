from gc_storage import list_files,get_img,get_report,get_negatives

# Get the list of all blobs (files)
files = list_files()
pos_files = [x for x in files if 'diagrams_with_google_label' in x.name]
neg_files = [x for x in files if 'negatives' in x.name]
# Note you could also do, but this is about twice as long
# pos_files = list_files('diagrams_with_google_label/')
# neg_files = list_files('negatives/')

# Get image and report from a given ID
report_id = '9120-2015-0083'
img = get_img(report_id,pos_files)
report = get_report(report_id,pos_files)

# Get negatives (won't work until all files are loaded)
negs = get_negatives(report_id,neg_files)
neg_img = get_img(negs,pos_files)
neg_report = get_report(negs,pos_files)