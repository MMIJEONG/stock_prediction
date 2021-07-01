#pip install flask
from flask import Flask,render_template
from prediction import prediction
app = Flask(__name__)

@app.route('/')
def test():
    return render_template('main.html')

if __name__ == '__main__':
    prediction()
    app.run(host='0.0.0.0',
            debug=True,use_reloader=False)