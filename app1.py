from flask import Flask, render_template, request
import joblib
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords
import spacy
import PIL
from keras.preprocessing import image
from keras.models import load_model

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

# ------------------------------------------------------------------ TEXTT MULAI DISINII --------------------------------------------------------------------
# Load flower data (for text-based prediction)
flower = pd.read_csv('DatasetBunga.csv')   #  WAJIB DI GANTI SESUAI TEMPAT NYIMPAN CSV NYA

# Create spaCy model and textcat component
nlp = spacy.blank("en")
textcat = nlp.add_pipe("textcat")

# Add labels to text classifier
textcat.add_label("Daisy")
textcat.add_label("Tulips")
textcat.add_label("Dandelion")
textcat.add_label("Roses")
textcat.add_label("Sunflower")

# Prepare training data
train_texts = flower['Text'].values
train_labels = [{'cats': {'Daisy': label == 'Daisy',
                          'Tulips': label == 'Tulips',
                          'Dandelion': label == 'Dandelion',
                          'Roses': label == 'Roses',
                          'Sunflower': label == 'Sunflower'}}
                for label in flower['Label']]
train_data = list(zip(train_texts, train_labels))

# Train the text model (if not already trained)
optimizer = nlp.begin_training()
for epoch in range(10):
    random.shuffle(train_data)
    losses = {}
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        nlp.update([example], sgd=optimizer, losses=losses)
    print(losses)

# Create a dictionary for label descriptions
label_descriptions = {
    "Daisy": "Bellis perennis adalah spesies Eropa yang umum dari daisy, dari familia Asteraceae, sering dianggap sebagai spesies archetypal dari nama itu. Bellis perennis asli dari Eropa barat, tengah dan utara, tetapi secara luas dinaturalisasi di sebagian besar daerah yang beriklim sedang termasuk Amerika dan Australasia. Secara umum, bunga daisy telah lama dikaitkan dengan simbol ketulusan, kemurnian, kelahiran kembali, dan keceriaan.",
    "Tulips": "Tulip (bahasa Latin: Tulipa) merupakan nama genus untuk 100 spesies tumbuhan berbunga yang termasuk ke dalam keluarga Liliaceae. Tulip berasal dari Asia Tengah, tumbuh liar di kawasan pegunungan Pamir dan pegunungan Hindu Kush dan stepa di Kazakhstan. Negeri Belanda terkenal sebagai negeri bunga tulip. Tulip juga merupakan bunga nasional Iran dan Turki.",
    "Dandelion": "Dandelion (Taraxacum) adalah genus besar dalam keluarga Asteraceae. Nama Randa Tapak sendiri biasanya merujuk kepada sebuah tumbuhan yang memiliki 'bunga-bunga' kecil yang dapat terbang ketika tertiup angin. Asal asli dari tumbuhan ini adalah Eropa dan Asia, tetapi sudah menyebar ke segala tempat. Yang disebut sebagai 'bunga' dari tumbuhan ini dapat menjadi semacam jam hayati yang secara teratur melepaskan banyak bijinya. Biji-biji ini sesungguhnya adalah buahnya.",
    "Roses": "Roses / rose adalah tumbuhan perdu, pohonnya berduri, bunganya berbau wangi dan berwarna indah, terdiri atas daun bunga yang bersusun:meliputi jenis, tumbuh tegak atau memanjat, batangnya berduri, bunganya beraneka warna, seperti merah, putih, merah jambu, merah tua, dan berbau harum. Mawar liar terdiri dari 100 spesies lebih, kebanyakan tumbuh di belahan bumi utara yang berudara sejuk. Spesies ini umumnya merupakan tanaman semak yang berduri atau tanaman memanjat yang tingginya bisa mencapai 2 sampai 5 meter. Walaupun jarang ditemui, tinggi tanaman mawar yang merambat di tanaman lain bisa mencapai 20 meter.",
    "Sunflower": "Bunga matahari (Helianthus annuus L.) adalah tumbuhan semusim dari suku kenikir-kenikiran (Asteraceae) yang populer, baik sebagai tanaman hias maupun tanaman penghasil minyak. Bunga tumbuhan ini sangat khas: besar, biasanya berwarna kuning terang, dengan kepala bunga yang besar (diameter bisa mencapai 30 cm). Bunga ini sebetulnya adalah bunga majemuk, tersusun dari ratusan hingga ribuan bunga kecil pada satu bongkol. Bunga Matahari juga memiliki perilaku khas, yaitu bunganya selalu menghadap / condong ke arah matahari atau heliotropisme. Orang Prancis menyebutnya tournesol atau 'Pengelana Matahari'. Namun, sifat ini disingkirkan pada berbagai kultivar baru untuk produksi minyak karena memakan banyak energi dan mengurangi hasil.",
}

# ------------------------------------------------------------------ TEXTT SELESAI DISINII --------------------------------------------------------------------

# ------------------------------------------------------------------ IMAGE MULAI DISINII --------------------------------------------------------------------
flower_names = {
    0: "Daisy",
    1: "Dandelion",
    2: "Roses",
}

# ------------------------------------------------------------------ IMAGE SELESAI DISINII --------------------------------------------------------------------

# Load image models
image_model = ResNet50()
sklearn_model = joblib.load("model.pkl")
modelGambar = load_model('model_flowers.h5')

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        docs = [nlp.tokenizer(text)]
        scores = textcat.predict(docs)
        prediction = scores.argmax(axis=1)[0]
        predicted_label = flower_names[prediction]
        description = label_descriptions.get(predicted_label, 'Description not available')

        image = PIL.Image.open(request.files['imagefile'].stream)
        img_convert = image.resize((256, 256))
        resized_img = img_to_array(img_convert) / 255
        resized_img = resized_img.reshape(1, 256, 256, 3)
        prediction = modelGambar.predict(resized_img)
        y_pred_labels = np.argmax(prediction, axis=1)[0]
        prediction_image = flower_names[y_pred_labels]
        description_image = label_descriptions.get(prediction_image, 'Description not available')

        return render_template("index.html", prediction=predicted_label, description=description, prediction_image=prediction_image, description_image=description_image)

if __name__ == "__main__":
    app.run(debug=True)