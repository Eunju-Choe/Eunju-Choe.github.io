---
title: 한글 URL 인코딩(+URL로 이미지 출력)
author: Eunju Choe
category: Python
tags: [Python, openapi]
img: ":20230511_2.png"
date: 2023-05-11 16:43:00 +0900
---
<p align="center"><img src="https://eunju-choe.github.io/assets/img/posts/20230511_2.png" width='50%'></p>

OpenAPI를 요청하기 위해 URL을 만드는 과정에서 한글을 인코딩하지 않아 오류가 발생하였습니다. 이번 글에서는 Python에서 한글을 인코딩하는 방법에 대해 알아보도록 하겠습니다. 던전앤파이터 API를 예시로 글을 작성하였습니다.

## 한글 URL 인코딩 방법


```python
from urllib.request import urlopen

# API Key 지정
key = "API_KEY"

serverId = 'casillas'
characterName = '대연동짱주먹'

url = f'https://api.neople.co.kr/df/servers/{serverId}/characters?characterName={characterName}&apikey={key}'

result_html = urlopen(url).read()
```


    ---------------------------------------------------------------------------

    UnicodeEncodeError                        Traceback (most recent call last)

    <ipython-input-23-c86ec08ff714> in <module>
          9 url = f'https://api.neople.co.kr/df/servers/{serverId}/characters?characterName={characterName}&apikey={key}'
         10 
    ---> 11 result_html = urlopen(url).read()
    

    /usr/lib/python3.6/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        221     else:
        222         opener = _opener
    --> 223     return opener.open(url, data, timeout)
        224 
        225 def install_opener(opener):


    /usr/lib/python3.6/urllib/request.py in open(self, fullurl, data, timeout)
        524             req = meth(req)
        525 
    --> 526         response = self._open(req, data)
        527 
        528         # post-process response


    /usr/lib/python3.6/urllib/request.py in _open(self, req, data)
        542         protocol = req.type
        543         result = self._call_chain(self.handle_open, protocol, protocol +
    --> 544                                   '_open', req)
        545         if result:
        546             return result


    /usr/lib/python3.6/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
        502         for handler in handlers:
        503             func = getattr(handler, meth_name)
    --> 504             result = func(*args)
        505             if result is not None:
        506                 return result


    /usr/lib/python3.6/urllib/request.py in https_open(self, req)
       1366         def https_open(self, req):
       1367             return self.do_open(http.client.HTTPSConnection, req,
    -> 1368                 context=self._context, check_hostname=self._check_hostname)
       1369 
       1370         https_request = AbstractHTTPHandler.do_request_


    /usr/lib/python3.6/urllib/request.py in do_open(self, http_class, req, **http_conn_args)
       1323             try:
       1324                 h.request(req.get_method(), req.selector, req.data, headers,
    -> 1325                           encode_chunked=req.has_header('Transfer-encoding'))
       1326             except OSError as err: # timeout error
       1327                 raise URLError(err)


    /usr/lib/python3.6/http/client.py in request(self, method, url, body, headers, encode_chunked)
       1283                 encode_chunked=False):
       1284         """Send a complete request to the server."""
    -> 1285         self._send_request(method, url, body, headers, encode_chunked)
       1286 
       1287     def _send_request(self, method, url, body, headers, encode_chunked):


    /usr/lib/python3.6/http/client.py in _send_request(self, method, url, body, headers, encode_chunked)
       1294             skips['skip_accept_encoding'] = 1
       1295 
    -> 1296         self.putrequest(method, url, **skips)
       1297 
       1298         # chunked encoding will happen if HTTP/1.1 is used and either


    /usr/lib/python3.6/http/client.py in putrequest(self, method, url, skip_host, skip_accept_encoding)
       1142 
       1143         # Non-ASCII characters should have been eliminated earlier
    -> 1144         self._output(request.encode('ascii'))
       1145 
       1146         if self._http_vsn == 11:


    UnicodeEncodeError: 'ascii' codec can't encode characters in position 50-55: ordinal not in range(128)


에러 메시지 "UnicodeEncodeError: 'ascii' codec can't encode characters in position 50-52: ordinal not in range(128)"는 일반적으로 URL에 한글 또는 특수문자와 같은 ASCII 이외의 문자가 포함되어 있을 때 발생합니다. URL은 ASCII 문자셋으로만 구성되어야 하며, 다른 문자들은 URL 인코딩을 통해 ASCII 문자로 변환되어야 합니다.

이러한 문제를 해결하기 위해서는 파이썬의 urllib 라이브러리를 사용하여 URL을 인코딩해야 합니다. urllib.parse 모듈의 quote() 함수를 사용하면 URL에 포함된 한글이나 특수문자를 인코딩할 수 있습니다.


```python
from urllib.parse import quote
import json

characterName = quote(characterName)
url = f'https://api.neople.co.kr/df/servers/{serverId}/characters?characterName={characterName}&apikey={key}'

print(characterName, end='\n\n')
result_html = urlopen(url).read()
result_json = json.loads(result_html)
print(json.dumps(result_json, indent=4))
```

    %EB%8C%80%EC%97%B0%EB%8F%99%EC%A7%B1%EC%A3%BC%EB%A8%B9
    
    {
        "rows": [
            {
                "serverId": "casillas",
                "characterId": "33f72784d2e6a522fc67134772293c1b",
                "characterName": "\ub300\uc5f0\ub3d9\uc9f1\uc8fc\uba39",
                "level": 14,
                "jobId": "1645c45aabb008c98406b3a16447040d",
                "jobGrowId": "4a1459a4fa3c7f59b6da2e43382ed0b9",
                "jobName": "\uadc0\uac80\uc0ac(\uc5ec)",
                "jobGrowName": "\ubca0\uac00\ubcf8\ub4dc"
            }
        ]
    }


첫 번째 줄은 quote를 이용하여 인코딩한 결과이며, 이제 정상적으로 호출되는 모습을 확인할 수 있습니다. 


```python
print(json.dumps(result_json, indent=4, ensure_ascii=False))
```

    {
        "rows": [
            {
                "serverId": "casillas",
                "characterId": "33f72784d2e6a522fc67134772293c1b",
                "characterName": "대연동짱주먹",
                "level": 14,
                "jobId": "1645c45aabb008c98406b3a16447040d",
                "jobGrowId": "4a1459a4fa3c7f59b6da2e43382ed0b9",
                "jobName": "귀검사(여)",
                "jobGrowName": "베가본드"
            }
        ]
    }


또한 다음과 같이 json.dumps() 함수의 ensure_ascii 매개변수를 False로 설정하여, 출력할 때 유니코드 문자를 그대로 출력하도록 지정할 수 있습니다.

## URL로 이미지를 출력하는 방법

추가로 파이썬으로 URL에서 이미지를 가져와 주피터 노트북에서 출력하는 방법에 대해 간단하게 알아보도록 하겠습니다.


```python
import requests
from PIL import Image
from io import BytesIO
import IPython.display as display

characterId = result_json['rows'][0]['characterId']
zoom=1

url = f'https://img-api.neople.co.kr/df/servers/{serverId}/characters/{characterId}?zoom={zoom}'

response = requests.get(url)
image = Image.open(BytesIO(response.content))
display.display(image)
```


    
![png](https://eunju-choe.github.io/assets/img/posts/20230511/output_10_0.png)
    


우선 이미지를 가져오기 위해 requests 라이브러리를 사용하며, 이미지를 처리하기 위해 PIL 라이브러리를 불러옵니다. requests.get() 함수를 사용하여 지정된 URL에서 이미지 데이터를 가져옵니다. 그리고 PIL의 Image.open() 함수를 사용하여 이미지를 엽니다. 마지막으로 IPython.display 모듈의 display() 함수를 사용하여 이미지를 주피터 노트북에서 출력합니다.

---

오늘은 파이썬에서 한글 URL을 인코딩하는 방법과 URL로 이미지를 출력하는 방법에 대해 알아보았습니다. 오늘은 이만 글 여기서 마치도록 하겠습니다.

읽어주셔서 감사합니다 :)
