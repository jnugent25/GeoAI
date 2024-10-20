import json, requests, csv,sys
from geopy.geocoders import Nominatim
size='640x640'
heading= '0'
pitch = '0'
key = 'AIzaSyD9XtE17j-M0cO0d950l_4B117XqNiGXlI'

geolocator = Nominatim(user_agent="MyApp")

with open(r"C:\Users\Jack\Downloads\Generated map (66654 locations).csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    data_read = [row for row in reader][10282:-9600]
for a,b,c in data_read:
    locationlat=a
    loclong=b
    location = geolocator.reverse(str(locationlat)+', '+str(loclong))

    address = location.raw['address']
    country = address.get('country_code', '').upper()
    print(country)
    r= requests.get(f'https://maps.googleapis.com/maps/api/streetview?size={size}&location={locationlat},{loclong}&heading={heading}&pitch={pitch}&key={key}')
    if r.status_code==200:
        with open(fr"C:\Users\Jack\Downloads\data\train\{country}\{locationlat+','+loclong}.jpg", 'wb') as f:
            f.write(r.content)
        with open(fr"C:\Users\Jack\Downloads\data\val\{country}\{locationlat+','+loclong}.jpg", 'wb') as f:
            f.write(r.content)
    r.close()

