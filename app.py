from flask import Flask,render_template,request,flash
from judge.classify_api import classify
app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    # request:请求对象-->获取请求方式 数据
    #1.判断请求方式
    if request.method=='POST':
        #2.获取请求的参数
        evaluation=request.form.get('evaluation')
        if(classify(evaluation)>0.5):
            print('这是一个积极评论')
        else:
            print('这是一个消极评论')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
