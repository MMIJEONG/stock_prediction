import requests
from bs4 import BeautifulSoup
import traceback
import pandas as pd
import datetime
import os

#마지막 페이지 번호 내에서 원하는 페이지의 테이블을 읽어옴
#parse_page() 함수는 종목과 페이지 번호를 입력으로 받아서 일별 주가를 Pandas DataFrame 객체로 반환
def parse_page(code, page):
    try:
        url = 'http://finance.naver.com/item/sise_day.nhn?code={code}&page={page}'.format(code=code, page=page)
        res = requests.get(url)
        _soap = BeautifulSoup(res.text, 'lxml')
        _df = pd.read_html(str(_soap.find("table")), header=0)[0]
        _df = _df.dropna()
        return _df
    except Exception as e:
        traceback.print_exc()
    return None

code = '016380'  # NAVER금융정보종목코드

#크롤링대상 url준비,request로 호출
url = 'https://finance.naver.com/item/sise_day.nhn?code={code}'.format(code=code)
res = requests.get(url)
res.encoding = 'utf-8'

#BeautifulSoup의 인스턴스를 생성
soap = BeautifulSoup(res.text, 'lxml')

#마지막페이지번호알아내
el_table_navi = soap.find("table", class_="Nnavi")
el_td_last = el_table_navi.find("td", class_="pgRR")
pg_last = el_td_last.a.get('href').rsplit('&')[1]
pg_last = pg_last.split('=')[1]
pg_last = int(pg_last)



#기준일자 설정
str_datefrom = datetime.datetime.strftime(datetime.datetime(year=2021, month=4, day=8), '%Y.%m.%d')
str_dateto = datetime.datetime.strftime(datetime.datetime.today(), '%Y.%m.%d')

#기준 일자 이후만 가져오는 로직
df = None
for page in range(1, pg_last+1):
    _df = parse_page(code, page)
    _df_filtered = _df[_df['날짜'] > str_datefrom]
    if df is None:
        df = _df_filtered
    else:
        df = pd.concat([df, _df_filtered])
    if len(_df) > len(_df_filtered):
        break

#DataFrame 객체를 CSV 파일로 저장
path_dir = '/Users/mijeong/Desktop/2021-04-08-crawling'
if not os.path.exists(path_dir):
    os.makedirs(path_dir)
path = os.path.join(path_dir, '{code}_{date_from}_{date_to}.csv'.format(code=code, date_from=str_datefrom, date_to=str_dateto))
#CSV 파일을 생성
df.to_csv(path, index=False)