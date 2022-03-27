from flask import Flask, render_template, request
import math
from identifikasi_idiom import IdiomIdentification
import identifikasi_idiom
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
        kal = kalimat_input
        hasil_identifikasi = hasil[0][2]

        if hasil_identifikasi == 'none':
            frasa_idiom = 'tidak terdapat idiom yang teridentifikasi'
        else:
            frasa_idiom = hasil_identifikasi

        hasil_arti = identifikasi_idiom.kamus_idiom(frasa_idiom)
        if len(hasil_arti) == 0:
            arti_idiom = 'arti dari idiom tidak ditemukan'
        else:
            kata, idiom, arti_idiom, contoh_kalimat = hasil_arti[0]

        if request.form['submit_button'] == 'Identifikasi':
            return render_template("identifikasi.html",kalimat=kal, idiom=frasa_idiom)
        elif request.form['submit_button'] == 'Cari Arti':
            return render_template("identifikasi.html",kalimat=kal, idiom=frasa_idiom, arti=arti_idiom)
    else:
        return render_template("identifikasi.html")


@app.route('/kamus-idiom', methods=['GET', 'POST'])
def kamus():
    ket = 'Hasil pencarian akan tampil disini'
    if request.method == 'POST':
        inputan = request.form['inputan']
        if request.form['submit_button'] == 'Cari Idiom':
            hasil = identifikasi_idiom.kamus_idiom(inputan)
            ket = 'Idiom tidak ditemukan dalam kamus'
            return render_template("kamus.html",hasil_idiom = hasil, input=inputan, keterangan = ket)
    else:
        return render_template("kamus.html", keterangan = ket)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    torch.multiprocessing.freeze_support()
    model = IdiomIdentification()
    app.run(debug=True)
