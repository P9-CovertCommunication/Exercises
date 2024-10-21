#################################################################################################
#                                                                                               #
# This is the property of the Authors, we gladly accept donations in the form of beer.          #
# Authors: Anders Bundgaard and Nicolai Lyholm                                                  #                                                   
# Date: 14/10/2024                                                                               #      
#                                                                                               #
#################################################################################################

import requests
import numpy as np

## POST requests
url_post = 'http://localhost:8080/signup'
requests.post(url_post,json={"name": ["Anders","Nicolai"]})

## ex 1 - Delete all numbers major than 30
print("==========EXERCISE 1==========")
url_ex = f'http://localhost:8080/ex/1'
print(url_ex)
r = requests.get(url_ex, headers={"x-data":"True"})
print(f"{r.json().keys()}")
data_json = r.json()['data']
print(f"Data before filtering: {data_json}")
data_under_30 = list(filter(lambda x: x<30, data_json))
print(f"Data after filtering {data_under_30}")
status = requests.post(url_ex,json={"data": data_under_30})
print(status)

## ex 2 - Sum all numbers to the last of the list (included
print("==========EXERCISE 2==========")

url_ex = f'http://localhost:8080/ex/2'
print(url_ex)
r = requests.get(url_ex, headers={"x-data":"True"})
print(f"{r.json().keys()}")
data_json = r.json()['data']
print(f"Data before processing: {data_json}")
answer = np.array(data_json)+data_json[-1]
print(f"The answer {answer}")
status = requests.post(url_ex,json={"data": answer.tolist()})
print(status)

## ex 3 - Sort strings in alphabethic order and then replace the first letter with 5
print("==========EXERCISE 3==========")
url_ex = f'http://localhost:8080/ex/3'
print(url_ex)
r = requests.get(url_ex, headers={"x-data":"True"})
print(f"{r.json().keys()}")
data_json = r.json()['data']
print(f"Data before processing: {data_json}")
data_jason_sorted = data_json.sort()
answer = ["5"+f"{x[1:]}" for x in data_json]
print(f"The answer {answer}")

status = requests.post(url_ex,json={"data": answer})
print(status)

## ex 4 - Repeat the string at *text* key how much mìhas indicated in the *n* key
print("==========EXERCISE 4==========")
url_ex = f'http://localhost:8080/ex/4'
print(url_ex)
r = requests.get(url_ex, headers={"x-data":"True"})
print(f"{r.json().keys()}")
data_json = r.json()['data']
print(f"Data before processing: {data_json}")
repeated_text = data_json['text']*data_json['n']
answer = repeated_text
print(f"The answer {answer}")
status = requests.post(url_ex,json={"data": answer})
print(status)

## ex 5 - Send a list containing the first 5 numbers divisible for the given number and major than 999
print("==========EXERCISE 5==========")
url_ex = f'http://localhost:8080/ex/5'
print(url_ex)
r = requests.get(url_ex, headers={"x-data":"True"})
print(f"{r.json().keys()}")
data_json = r.json()['data']
print(f"Data before processing: {data_json}")
answer = [x for x in range(1000,1000+6*data_json) if x%data_json==0][:5]
print(f"The answer {answer}")
status = requests.post(url_ex,json={"data": answer})
print(status)