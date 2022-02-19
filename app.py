from flask import Flask, render_template, request
import math
from identifikasi_idiom import IdiomIdentification
from identifikasi_idiom import TokenSimilarity
import torch
import logging
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/identifikasi-idiom', methods=['GET', 'POST'])
def identifikasi():
    if request.method == 'POST':
        kalimat_input = request.form['input_kalimat']
        hasil = model.predict([kalimat_input])
        kal = hasil[0][0]
        hasil_identifikasi = hasil[0][2]

        if hasil_identifikasi == 'none':
            frasa_idiom = 'Tidak terdapat idiom yang teridentifikasi'
        else:
            frasa_idiom = hasil_identifikasi

        arti_idiom = 'ini artinya'

        if request.form['submit_button'] == 'Identifikasi':
            return render_template("identifikasi.html",kalimat=kal, idiom=frasa_idiom)
        elif request.form['submit_button'] == 'Cari Arti':
            return render_template("identifikasi.html",kalimat=kal, idiom=frasa_idiom, arti=arti_idiom)
    else:
        return render_template("identifikasi.html")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    torch.multiprocessing.freeze_support()
    model = IdiomIdentification()
    app.run(debug=True)
