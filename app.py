from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello'


@app.route('/hola')
def hola():
    return 'hola'

if __name__ == '__main__':
    app.run()
