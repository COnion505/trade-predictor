# trade-predictor

tensorFlowを使用して為替の予測を行う。過去数年分のデータを取得してDBに保存、最新のデータは外部のAPI使用してDBに更新していく。

## 環境
- python3.9
- tensorFlow2.8(keras2.8,numpy1.22)
- matplotlib3.5
- pandas1.4.1
- sklearn1.0.2
- psycopg2(connect to postgres)

## set up pip

```terminal
python -m venv --system-site-packages .\venv
.\venv\Script\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## set up docker

```terminal
docker pull tensorflow/tensorflow:latest-gpu
docker run --gpus all -it --volume $PWD:/tmp -w /tmp tensorflow/tensorflow:latest-gpu python3 mnist_tutorial.py
```

## link
- https://recruit.gmo.jp/engineer/jisedai/blog/fx_lstm_with_technical_index/
- https://sinyblog.com/deaplearning/preprocessing_002/
- https://tmytokai.github.io/open-ed/activity/dlearning/text05/page02.html
- https://www.dukascopy.com/trading-tools/widgets/quotes/historical_data_feed

## 用語など

- バッチサイズ
  batch_size  
  勾配降下法でデータをいくつかのサブセットに分ける必要がある。そのサイズのこと。  
  1000件のデータを200件ずつのサブセットに分ける場合はbatch_sizeは200になる。  
  2のn乗が慣習的に使われる。  
- イテレーション
  iteration  
  バッチサイズが決まれば自動的に決まる。  
  先の例の場合、1000件を200件ずつのサブセットに分けているのでイテレーションは5（1000÷200）  
  データセットに含まれるデータが少なくとも一回は学習に用いられるのに必要な学習回数。  
- エポック数
  epoch  
  1.データセットをバッチサイズに従ってN個のサブセットに分ける。  
  2.各サブセットを学習に回す。つまり、N回学習を繰り返す。  
  1と2の手順により、データセットに含まれるデータは少なくとも1回は学習に用いられることになります。  
  そして、この1と2の手順を1回実行することを1エポックと呼びます。  
  損失関数の値がほぼ収束するまで繰り返すのが適正なエポック数。  

