import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}

# cifar-10画像のラベル
labels = ["飛行機","自動車", "鳥", "猫", "鹿", "犬", "カエル", "馬", "船", "トラック"]
n_class = len(labels)
img_size = 32 # 学習時のサイズに合わせる
n_result = 3  # 上位3つの結果を表示

# アプリ生成とコンフィギュレーションの設定
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# アップロードされた画像が許可された拡張子か判定
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

# アプリのルーティング、画面の描画
@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

# ファイルのアップロード時の操作
@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
        # ファイルの形式と存在を確認
        if "file" not in request.files:
            print("file doesn't exist.")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ":file not allowed")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        # ファイル名を安全なものにする
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        x = np.array(image, dtype=float)
        x = x.reshape(1, img_size, img_size, 3) / 255

        # 予測
        model = load_model("./image_classifier.h5")
        y = model.predict(x)[0]
        # 降順でソート
        sorted_idx = np.argsort(y)[::-1]
        result = ""
        for i in range(n_result):
            idx = sorted_idx[i]
            ratio = y[idx]
            label = labels[idx]
            result += "<p>" + str(round(ratio * 100, 1)) + "%の確立で" + label + "です。</p>"
        return render_template("result.html", result=Markup(result), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)