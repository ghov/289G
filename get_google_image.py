import csv
import io
import http.client
import const

read_file_path = '/home/greghovhannisyan/PycharmProjects/289G Project/batch_2.csv'
limit = 25000
counter = 0

def getpng(lat, lon, zoom):
    urlpath = (
        const.URLBASE +
        "?maptype=" + const.MAPTYPE +
        "&size=" + const.RESX + "x" + const.RESY +
        "&center=" + str(lat) + "," + str(lon) +
        "&zoom=" + str(zoom) +
        "&scale=2" +
        "&key=" + const.APIKEY
    )
    conn = http.client.HTTPSConnection(const.HOST)
    conn.request("GET", urlpath)
    resp = conn.getresponse()
    if (resp.status != 200):
        print(urlpath)
        print(resp.status, resp.reason)
        return None
    data = resp.read()
    return data

def savepng(pngdata, filename):
    with io.open(filename, 'wb') as f:
        f.write(pngdata)


with open(read_file_path, 'r') as csv_read_file:
    my_reader = csv.DictReader(csv_read_file)

    for row in my_reader:
        if(counter <= limit):
            counter +=1

            lat = float(row['LATITUDE'])
            lon = float(row['LONGITUDE'])
            zoom = 19

            pngdata = getpng(lat, lon, zoom)
            if (pngdata is None):
                print("error")
            else:
                savepng(pngdata, '/home/greghovhannisyan/Desktop/Json_files/google_images_batch_2/' + row['REPORTNUMBER'] + "_google.png")


