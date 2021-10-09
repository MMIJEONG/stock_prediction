import pymysql
from operator import itemgetter

def m_stock_item():
    # db연동
    stock_db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='',#자신비밀번호
        db='stock',
        charset='utf8'
    )
    cursor = stock_db.cursor(pymysql.cursors.DictCursor)
    sql="SELECT * FROM stock_item;"
    cursor.execute(sql)
    result = cursor.fetchall()
    #print(result)
    result = sorted(result, key=itemgetter('stock_name'))  # 종목들을 이름순으로 정렬
    return result#모든 주식이름,주식코드를 return

    stock_db.close()