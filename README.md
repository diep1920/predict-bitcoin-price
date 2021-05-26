# Tài nguyên:

+ Kraken API

###  Mã nguồn:
+ crawl_data.py: lấy dữ liệu từ KrakenAPI.
+ params.txt: tham số cho crawl_data.py với "pairs" là mã cặp asset, ở đây là Bitcoin-USD(XBT-USD), "interval" là chu kỳ lấy mẫu, "no_of_records" là số lượng bản ghi, "name": là tên file .csv lưu dữ liệu lấy về.
+ prepare_dataset.py: đưa dữ liệu về dạng phù hợp vào đầu vào mô hình, chia dữ liệu thành tập train và tập test.
+ tune_hyperparams.py: chỉnh siêu tham số cho các mô hình. Do không đủ tài nguyên tính toán, project này chỉ chỉnh số lượng layer, có thể chỉnh thêm tham số khác, có hướng dẫn trong mã nguồn. 
+ train_and_test.py: huấn luyện và đánh giá mô hình.
+ models/conv1d.py, lstm.py, gru.py, và rnn.py: các mô hình đề xuất Conv1DNet, LSTMNet, GRUNet, và SimpleRNNNet.
+ figure/ : lưu các đồ thị validation, đồ thị so sánh giá trị ask_price/bid_price dự đoán với giá trị thức, bảng tham chiếu bộ siêu tham số


# Step-by-step:

## 0. Các thư viện
+ Cài pykrakenapi theo hướng dẫn tại https://github.com/dominiktraxl/pykrakenapi
+ keras, sklearn, numpy, pandas, matplotlib

## 1. Lấy dữ liệu (tuỳ chọn)
Chạy
```sh
python crawl_data.py
```
Tham số điều chỉnh ở file params.txt

## 2. Chuẩn bị dữ liệu
Chạy 

```sh
python prepare_dataset.py
```

## 3. Tìm siêu tham số (tuỳ chọn)
Chạy 

```sh
python tune_hyperparams.py
```
Quan sát kết quả ở các đồ thị validation, đối chiếu với bảng tham chiếu và chọn bộ tham số theo ý muốn.


## 4. Huấn luyện và đánh giá
Chạy 

```sh
python train_and_test.py 0
```
tham số là chỉ số của mô hình. 0 cho LSTMNet, 1 cho GRUNet, 2 cho SimpleRNNNet, 3 cho Conv1DNet
Quan sát kết quả ở các đồ thị ask_price_prediction và bid_price_prediction


