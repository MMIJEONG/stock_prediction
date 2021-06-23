from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd

#chromedriver의 위치지정
driver = webdriver.Chrome('/Users/mijeong/Downloads/chromedriver')
# 암묵적으로 웹 자원 로드를 위해 3초 대기
driver.implicitly_wait(3)

def print_stock_price(code):
    col=[]
    result = [[],[],[],[],[],[]]
    n=0
    while True:
        url = 'https://finance.naver.com/item/sise_day.nhn?code='+code+'&page='+str(n+1) # url에 접근한다.
        driver.get(url)
        html = driver.page_source # 페이지의 elements모두 가져오기
        soup = BeautifulSoup(html, 'html.parser')# BeautifulSoup사용하기
        tr = soup.select('body > table.type2 > tbody > tr')
        if n==0 :#첫번째페이지크롤링때 마지막페이지수 구함
            a=soup.select('body > table.Nnavi > tbody > tr > td.pgRR > a')
            pagenum=int(a[0]["href"].split("page=")[1])

        for i in range(1, len(tr)-1):#각페이지에 있는 데이터들 수집
            if tr[i].select('td')[0].text.strip():
                result[0].append(tr[i].select('td')[0].text.strip().replace(".",""))
                result[1].append(tr[i].select('td')[1].text.strip().replace(",",""))
                result[2].append(tr[i].select('td')[3].text.strip().replace(",",""))
                result[3].append(tr[i].select('td')[4].text.strip().replace(",",""))
                result[4].append(tr[i].select('td')[5].text.strip().replace(",",""))
                result[5].append(tr[i].select('td')[6].text.strip().replace(",",""))
        n+=1
        time.sleep(2)#크롤링 시간지연주기
        #if n==3:#테스트를 위한 페이지수
            #break
        if n==pagenum: #마지막페이지면 break
           break


    col.append(tr[0].select('th')[0].text.strip())#항목명 구함
    col.append(tr[0].select('th')[1].text.strip())
    col.append(tr[0].select('th')[3].text.strip())
    col.append(tr[0].select('th')[4].text.strip())
    col.append(tr[0].select('th')[5].text.strip())
    col.append(tr[0].select('th')[6].text.strip())

    #terminal print
    '''for i in range(6):
        if i==5:
            print(col[i],end="\n")
            break
        print(col[i],end=" ")'''

    '''for i in range(len(result[0])):
        print(result[0][i], result[1][i], result[2][i],result[3][i],result[4][i],result[5][i])'''

    #csv파일로 출력
    df = pd.DataFrame(
        {
            '날짜':result[0],
            '시가':result[2],
            '고가':result[3],
            '저가':result[4],
            '종가':result[1],
            '거래량':result[5]
        }
    )
    df.set_index('날짜', inplace=True)
    df.to_csv("./삼성전자.csv")


stock_code = '005930'#중목코드

print_stock_price(stock_code)

