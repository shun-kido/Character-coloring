# Character-coloring
機械学習初心者がpix2pixを使ってどこまで線画を着色できるか  
一応web上で遊べるのを目指す
![fireshot capture 9 - coloringai - http___127 0 0 1_5000_seznd](https://user-images.githubusercontent.com/45202725/52843549-c9d20700-3145-11e9-88b8-19417b8b5f70.png)

# Requirements
・Python3  
・tensorflow(-gpu)+keras  
・Numpy  
・openCV  
・flask

# Dataset
128×128にリサイズしたイラストデータと抽出した線画23000組
前処理（make_dataset.py）  
・lbpcascade_animeface.xmlを用いて顔検知、Trueのみをデータとして使用  
・HSVに変換し、Sの閾値を設定し白黒画像がデータに入らないように（コード内ではs<18を除外）  
・128×128にリサイズしたイラストから線画を抽出  
・抽出した線画に元イラストの部分的な色情報をヒント(5~15個)を与える  

# train
pix2pixを使って学習
model:https://github.com/tommyfms2/pix2pix-keras-byt

# example
server.pyを実行するとweb上でヒントを付け着色してくれるサイトが動きます
画像を選択するとこのような画面になります
![fireshot capture 10 - coloringai - http___127 0 0 1_5000_send](https://user-images.githubusercontent.com/45202725/52842716-4fa08300-3143-11e9-92e5-f1c04a8d6319.png)  
↓　web上でヒントを描くことができます　　
![fireshot capture 8 - coloringai - http___127 0 0 1_5000_](https://user-images.githubusercontent.com/45202725/52843572-dd7d6d80-3145-11e9-9230-170921214638.png)　　
↓　着色する！を押すと着色前と着色後の画像が表示されます（画像サイズは小さいです）  
![fireshot capture 9 - coloringai - http___127 0 0 1_5000_send](https://user-images.githubusercontent.com/45202725/52843743-54b30180-3146-11e9-8cdf-05fe9daf6d10.png)
