import pymysql

def m_recommend():
    result_arr=[]
    # db연동
    stock_db = pymysql.connect(
        user='root',
        passwd='password',
        host='127.0.0.1',  # localhost
        db='stock',
        charset='utf8'
    )
    cursor = stock_db.cursor(pymysql.cursors.DictCursor)
    sql="SELECT name,percent FROM stock_info ORDER BY percent DESC;"
    cursor.execute(sql)
    for i in range(3):
        result = cursor.fetchone()
        result_arr.append(result)
    #print(result_arr)
    return result_arr

    stock_db.close()