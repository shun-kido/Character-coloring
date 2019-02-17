# Character-coloring
ヒントを基にイラストを着色してくれるAIとそれを使ったwebサービス
![fireshot capture 9 - coloringai - http___127 0 0 1_5000_seznd](https://user-images.githubusercontent.com/45202725/52843549-c9d20700-3145-11e9-88b8-19417b8b5f70.png)

# 作成理由
・画像分野の機械学習について実際に物を作りながら触れたかった  
・アプリを作るというビジョンが見えなかったため、一通り自分で作成してみたかった  
・既存開発であるPaintChainerに感動し、違うアプローチから似たものを自身の手で作成出来たら楽しいだろうと感じたから  

# Dataset
スクレイピングしたイラストを128×128にリサイズしたイラストデータと抽出した線画23000組  
前処理（make_dataset.py）  
・lbpcascade_animeface.xmlを用いて顔検知、Trueのみをデータとして使用  
・HSVに変換し、Sの閾値を設定し白黒画像がデータに入らないように（コード内ではs<18を除外）  
・128×128にリサイズしたイラストから線画を抽出  
・抽出した線画に元イラストの部分的な色情報をヒント(5~15個)を与える  

# train
pix2pixを使って学習  
modelは既存のものをお借りしました

# example
画像を選択するとこのような画面になります  
![fireshot capture 10 - coloringai - http___127 0 0 1_5000_send](https://user-images.githubusercontent.com/45202725/52842716-4fa08300-3143-11e9-92e5-f1c04a8d6319.png)  
↓　web上でヒントを描くことができます　　
![fireshot capture 8 - coloringai - http___127 0 0 1_5000_](https://user-images.githubusercontent.com/45202725/52843572-dd7d6d80-3145-11e9-9230-170921214638.png)　　
↓　着色する！を押すと着色前と着色後の画像が表示されます（画像サイズは小さいです）  
![fireshot capture 9 - coloringai - http___127 0 0 1_5000_send](https://user-images.githubusercontent.com/45202725/52843743-54b30180-3146-11e9-8cdf-05fe9daf6d10.png)


使用言語:Python, HTML, CSS, JavaScript  
開発環境:ubuntu, flask, tensorflow, keras  
制作期間：2018年12月~
