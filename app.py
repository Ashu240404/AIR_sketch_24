from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/start')
def start_camera():
    subprocess.Popen(['python', 'mycam.py'], shell=True)
    return 'Camera started! Check the Python window.'

if __name__ == '__main__':
    app.run(debug=True)
