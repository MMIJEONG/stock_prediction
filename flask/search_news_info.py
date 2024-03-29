import pymysql
import urllib.request
import json
from urllib import parse

def search_news_info(info):
    info = info.getlist('news')
    client_id = ""#네이버부여해준ID
    client_secret = ""#네이버부여해준비번
    encText = urllib.parse.quote(info[0])  # 검색할 키워드
    display = 10  # 검색 결과 출력 건수 지정을 10개로 지정
    start = 1  # 검색 시작위치를 1으로 지정
    sort = "sim"  # 정렬 기준을 유사도순으로 지정
    # f-string을 사용해서 요청 변수들 적용
    before_url = f"https://openapi.naver.com/v1/search/news?query={encText}&display={display}&start={start}&sort={sort}"
    url = parse.urlparse(before_url)
    query = parse.parse_qs(url.query)
    query = parse.urlencode(query, doseq=True)
    url = f"https://openapi.naver.com/v1/search/news?{query}"
    #url = "https://openapi.naver.com/v1/search/news?query=" + encText #json으로 결과 받기
    #url = "https://openapi.naver.com/v1/search/news.xml?query=" + encText #xml로 결과 받기
    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(request)
    rescode = response.getcode()
    if (rescode == 200):
        response_body = response.read()
        #print(response_body.decode('utf-8'))
        return json.loads(response_body.decode('utf-8'))
    else:
        print("Error Code:" + rescode)