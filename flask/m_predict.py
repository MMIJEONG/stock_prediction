import pymysql

def m_predict(name):
    stock_name=name
    # db연동
    stock_db = pymysql.connect(
        user='root',
        passwd='password',
        host='127.0.0.1',  # localhost
        db='stock',
        charset='utf8'
    )
    cursor = stock_db.cursor(pymysql.cursors.DictCursor)
    sql="SELECT * FROM stock_info WHERE name=%s;"
    cursor.execute(sql,stock_name)
    result = [cursor.fetchone()]
    #print(result)
    return result

    stock_db.close()