import requests
import json


# url = 'http://localhost:5000/predict'
# r = requests.post(url,json={'txt':'أضئ المصباح',})
# print(r.text)


from requests import get

ip = get('https://sameh-m-amin-vd2ocl599b9z0yop.socketxp.com:5000/predict').text
print('My public IP address is: {}'.format(ip))