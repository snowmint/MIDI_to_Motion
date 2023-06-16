# MIDI_to_Motion

## 1. data_preprocess.py
(原則上不用執行。除非有修改 code ，否則我已經運行過，並儲存 pickle file 在指定目錄下了。)

python data_preprocess.py 

## 2. train.ipynb
包含訓練和測試的程式，超參數也定義於此。

## 3. model.py
這份程式實作了 LSTM encoder-decoder 模型。

## 4. data_loader.py
定義 PyTorch Data loader 如何取用訓練資料。

Dataset:

`init function`: 只給予訓練資料 pickle file 的路徑。

`get_item function`: 在指定的訓練資料路徑讀取 pickle file，並隨機在歌曲內挑選長度為 512 的片段。只在樂曲的開頭加上全為 0 的 <start-of-token>，以及在樂曲結尾加上全為 1 的 <end-of-token> 。
  
`len function`: 由於目前一首歌曲算一筆資料，透過設定 dataset 需要 100 倍的資料量，便可以使每一首歌都會隨機取用 100 個隨機片段。

## 5. test result
100 epoch 訓練後的測試結果：
[100epoch]22min_(custom_loss)random_pick_2200_datasample_per_epoch
https://drive.google.com/drive/folders/1SobWLwwDAmP6CrF-iHaoQJWU0ozB4pq6?usp=drive_link
