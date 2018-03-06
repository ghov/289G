import csv
import json
import random

# TO CHANGE
# file path 1, 2 and 3
# change GOOGLE_APPLICATION_CREDENTIALS file path
# must include the TSN_export_289G.csv file in the same folder

read_file_path = '/Users/Linen/GoogleDrive/ECS289G/final_project/TSN_export_289G.csv'
write_file_path2 = '/Users/Linen/Downloads/negatives_diffCountyAndRoute/'
write_file_path3 = '/Users/Linen/Downloads/negatives_no_constraint/'
# 100 hard negatives for each report ID
# 'a'
from matplotlib.image import imread
from google.cloud import storage
import numpy as np
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/Linen/GoogleDrive/GitHub/289G/StorageClient/gc_storage_auth.json"

client = storage.Client()
bucket_name = 'sample_report_images'
bucket = client.get_bucket(bucket_name)


dirDict = {}
counter = 0
# counter2 = 0
for x in bucket.list_blobs():
    if ('diagrams_with_google_label/' in x.name):
        dirName = x.name.split('/')[1]
        # print(x.name.split('/'))
        dirDict[dirName] = dirDict.get(dirName, 0) + 1
    #     counter2 += 1
    # if counter2 > 10:
    #     break
print(len(dirDict))
# dirDict["testing"] = 4
output = {}
for k, v in dirDict.items():
    if v ==  2:
        counter += 1
        if output.get(v, 99) == 99:
            output[v] = [k,]
        else:
            output[v].append(k)

print(counter)

output['current_total'] = len(dirDict)
output['report_with_single_diagram_count'] = counter


write_file_path = '/Users/Linen/GoogleDrive/ECS289G/final_project/current_single_diagram_reports.json'
with open(write_file_path, 'w+') as outfile:
    json.dump(output, outfile)




# dirDict = {}
#
# for x in bucket.list_blobs():
#     if ('diagrams_with_google_label/' in x.name):
#         dirName = x.name.split('/')[1]
#         dirDict[dirName] = 0
dirList = output[2]
print(len(dirList))


with open(read_file_path, 'r', encoding = 'ISO-8859-1') as csvfile:
    reader = csv.DictReader(csvfile)
    reportList = []
    data = {}


    for row in reader:
        rptNum = row['REPORTNUMBER']
        county = row['SWI_COUNTY']
        if row['SWI_ROUTE'] != '':
            route = int(float(row['SWI_ROUTE']))
        else:
            route = 0
        reportList.append(rptNum)
        data[rptNum] = [county, route]

size = len(dirList)
count = 0
for report in dirList:
    output2 = {}
    output2[report] = []
    counter = 1

    while counter <= 100:
        r = random.randint(0,size-1)
        randReportNum = dirList[r]
        if report != randReportNum and data[report][0] != data[randReportNum][0] and data[report][1] != data[randReportNum][1] and data[randReportNum][0] != '' and data[randReportNum][1] != 0:
            output2[report].append(randReportNum)
            counter += 1
        # if report != randReportNum and data[randReportNum][0] != '' and data[randReportNum][1] != 0:
        #     output2[report].append(randReportNum)
        #     counter += 1

    with open(write_file_path2 + report + '_negative.json', 'w+') as outfile2:
        json.dump(output2, outfile2)

for report3 in dirList:
    output3 = {}
    output3[report3] = []
    counter = 1

    while counter <= 100:
        r = random.randint(0,size-1)
        randReportNum = dirList[r]
        if report3 != randReportNum and data[randReportNum][0] != '' and data[randReportNum][1] != 0:
            output3[report3].append(randReportNum)
            counter += 1

    with open(write_file_path3 + report3 + '_negative.json', 'w+') as outfile3:
        json.dump(output3, outfile3)

        # data = {}
        # data[rptNum] = []
        # counter = 0
        # while counter < 100:
        #     randomID = random.randint(0,252527)
        #     # if the random row contains different county
        #     row['']
        #     data[rptNum].append()




# with open(write_file_path, 'w') as jsonfile:
