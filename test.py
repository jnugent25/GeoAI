import csv
with open(r"C:\Users\Jack\Downloads\Generated map (66654 locations).csv") as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # next(reader, None)  # skip the headers
    data_read = [row for row in reader][:-9600]
    print(len(data_read))
#print(data_read)

import requests
import urllib3
#r=requests.get('http://api.geonames.org/countryCodeJSON?lat=49.03&lng=10.2&username=demo')


from geopy.geocoders import Nominatim

# Initialize Nominatim API
geolocator = Nominatim(user_agent="MyApp")
countries=[]
print('test')
for i,(a,b,c) in enumerate(data_read):
# Latitude & Longitude input
    locationlat=a
    loclong=b
    print(i)
    location = geolocator.reverse(str(locationlat)+', '+str(loclong))

    address = location.raw['address']

    country = address.get('country_code', '')
    countries.append(country)
print(countries)