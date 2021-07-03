#pip install flask
from flask import Flask,render_template
from prediction import prediction
from m_recommend import m_recommend
from m_predict import m_predict
from m_stock_item import m_stock_item
app = Flask(__name__)

@app.route('/')
def main_page():
    return render_template('main.html')

@app.route('/predict')
def predict_page():
    item_arr=m_stock_item()
    return render_template('predict.html',result=item_arr)

@app.route('/predict/<target>')
def predict_print_page(target):
    predict_result=m_predict(target) #주식예측결과리턴
    print(predict_result)
    return render_template('predict_print.html',result=predict_result[0])

@app.route('/recommend')
def recommend_page():
    rec_arr = m_recommend() #주식종목추천 모듈
    return render_template('recommend.html',result=rec_arr)

if __name__ == '__main__':
    #prediction() 주식예측실행 예측실행전 stock_item 디비에 주식종목이름,코드가 저장되어있어야함!
    app.run(host='0.0.0.0',
            debug=True,use_reloader=False)