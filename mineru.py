import requests

url='https://mineru.net/api/v4/extract/task'
header = {
    'Content-Type':'application/json',
    "Authorization":"Bearer eyJ0eXBlIjoiSl...请填写准确的token！"
}
data = {
    'url':'https://cdn-mineru.openxlab.org.cn/demo/example.pdf',
    'is_ocr':True,
    'enable_formula': False,
}

res = requests.post(url,headers=header,json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])