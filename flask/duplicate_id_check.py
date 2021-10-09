import pymysql

def duplicate_id_check(info):
    #db연동
    stock_db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='',#자신비밀번호
        db='stock',
        charset='utf8'
    )
    cursor = stock_db.cursor(pymysql.cursors.DictCursor)
    sql = "SELECT id FROM members WHERE id=%s;"  #쿼리를 사용해 중복을 체크
    cursor.execute(sql,(info['checkid']))
    result = cursor.fetchall()
    stock_db.close()

    if result:
        return False #중복
    else:
        return True #중복아님

