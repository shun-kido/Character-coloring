# Character-coloring
機械学習初心者がpix2pixを使ってどこまで線画を着色できるか
一応web上で遊べるのを目指す
images.githubusercontent.com/45202725/52842716-4fa08300-3143-11e9-92e5-f1c04a8d6319.png)

# Requirements
・Python3  
・tensorflow(-gpu)+keras  
・Numpy  
・openCV  

# Dataset
128×128にリサイズしたイラストデータと抽出した線画23000組
前処理（make_dataset.py）　　
・lbpcascade_animeface.xmlを用いて顔検知、Trueのみをデータとして使用　　
・HSVに変換し、Sの閾値を設定し白黒画像がデータに入らないように（コード内ではs<18を除外）　　
・128×128にリサイズしたイラストから線画を抽出　　
・抽出した線画に元イラストの部分的な色情報をヒント(5~15個)を与える　　

