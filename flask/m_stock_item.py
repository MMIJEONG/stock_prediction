import pymysql

def m_stock_item():
    # db연동
    stock_db = pymysql.connect(
        user='root',
        passwd='password',
        host='127.0.0.1',  # localhost
        db='stock',
        charset='utf8'
    )
    cursor = stock_db.cursor(pymysql.cursors.DictCursor)
    sql="SELECT * FROM stock_item;"
    cursor.execute(sql)
    result = cursor.fetchall()
    #print(result)
    return result

    stock_db.close()