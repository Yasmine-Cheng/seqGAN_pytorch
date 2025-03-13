#IMPORTING LIBRARIES AND MODULES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from nltk import word_tokenize, sent_tokenize
import gc
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from datetime import datetime

# 全局變量
word_ind = {}  # 字典，單詞到索引的映射
ind_word = {}  # 字典，索引到單詞的映射
seq_len = 16   # 序列長度

#Loading data
def load_data(filepath):
    """加載文本文件"""
    try:
        f = open(filepath, encoding='utf-8')
        return f.read()
    except UnicodeDecodeError:
        try:
            # 嘗試不同編碼
            f = open(filepath, encoding='cp950')
            return f.read()
        except:
            # 最後嘗試二進制讀取再解碼
            with open(filepath, 'rb') as f:
                content = f.read()
                return content.decode('utf-8', errors='ignore')

#Cleaning data
def Clean_data(data):
    """清理文本數據"""
    repl = '' 
    
    # 基本清理
    data = re.sub('\(', repl, data)
    data = re.sub('\)', repl, data)
    
    # 移除標題
    for pattern in set(re.findall("=.*=", data)):
        data = re.sub(pattern, repl, data)
    
    # 移除未知詞
    for pattern in set(re.findall("<unk>", data)):
        data = re.sub(pattern, repl, data)
    
    # 移除非字母數字字符
    for pattern in set(re.findall(r"[^\w ]", data)):
        repl = ''
        if pattern == '-':
            repl = ' '
        if pattern != '.' and pattern != "\'":
            try:
                data = re.sub("\\"+pattern, repl, data)
            except:
                pass
            
    return data

def split_data(data, num_sentences=-1):
    """分割文本為句子和詞，保留所有詞彙"""
    # 句子分割
    if num_sentences == -1:
        sentences = sent_tokenize(data)
    else:
        sentences = sent_tokenize(data)[:num_sentences]
    
    # 詞分割和詞頻統計
    word_freq = {}
    for sent in sentences:
        for word in str.split(sent, ' '):
            if word.strip():  # 只添加非空詞
                word_freq[word] = word_freq.get(word, 0) + 1
    
    # 創建詞彙表 - 保留所有詞，不做頻率過濾
    words = [""]  # 首先添加空字符串用於padding
    words.append("<UNK>")  # 添加未知詞標記
    
    # 添加所有詞
    for word in word_freq.keys():
        if word.strip():  # 確保不添加空詞
            words.append(word)
    
    print(f"詞彙表大小（原始）: {len(word_freq)}")
    print(f"詞彙表大小（加入特殊標記後）: {len(words)}")
    
    return sentences, words

def Convert_data(sentences, words, seq_len):
    """將文本轉換為數值形式，改進版：減少<UNK>的使用"""
    global word_ind, ind_word
    
    # 建立詞到索引的字典
    word_ind = {word: idx for idx, word in enumerate(words)}
    
    # 建立索引到詞的字典
    ind_word = {idx: word for idx, word in enumerate(words)}
    
    # 未知詞的索引
    unk_idx = word_ind.get("<UNK>", 0)
    pad_idx = word_ind.get("", 0)
    
    print(f"處理 {len(sentences)} 個句子...")
    
    sent_sequences = []
    total_words = 0
    unk_count = 0
    
    for i in range(len(sentences)):
        words_in_sent = str.split(sentences[i], ' ')
        words_in_sent = [w for w in words_in_sent if w.strip()]  # 過濾空詞
        
        # 處理句子中的每個詞，盡量保留原始詞彙
        processed_words = []
        for word in words_in_sent:
            total_words += 1
            if word in word_ind:
                processed_words.append(word)
            else:
                # 詞不在詞彙表中，計數並使用<UNK>
                unk_count += 1
                processed_words.append("<UNK>")
        
        # 跳過過短的句子
        if len(processed_words) <= 1:
            continue
            
        # 創建序列
        for j in range(1, len(processed_words)):
            if j <= seq_len:
                sent_sequences.append(processed_words[:j])
            elif j > seq_len and j < len(processed_words):
                sent_sequences.append(processed_words[j-seq_len:j])
            elif j > len(processed_words)-seq_len:
                sent_sequences.append(processed_words[j-seq_len:])
    
    print(f"詞彙表中詞總數: {len(words)}")
    print(f"文本中詞總數: {total_words}")
    print(f"使用<UNK>替換的詞數: {unk_count} ({unk_count/max(1,total_words)*100:.2f}%)")
    
    # 過濾掉過短的序列
    sent_sequences = [seq for seq in sent_sequences if len(seq) > 1]
    print(f"生成序列總數: {len(sent_sequences)}")
                
    # 分割為預測器和標籤
    predictors = []; class_labels = []
    for i in range(len(sent_sequences)):
        predictors.append(sent_sequences[i][:-1])
        class_labels.append(sent_sequences[i][-1])
    
    # 手動填充預測器
    pad_predictors = []
    for i in range(len(predictors)):
        emptypad = [''] * (seq_len-len(predictors[i])-1)
        emptypad.extend(predictors[i])
        pad_predictors.append(emptypad)
            
    # 將每個詞轉換為對應的索引
    for i in range(len(pad_predictors)):
        for j in range(len(pad_predictors[i])):
            word = pad_predictors[i][j]
            pad_predictors[i][j] = word_ind.get(word, unk_idx)  # 使用get避免KeyError
        
        # 標籤也轉換為索引
        class_labels[i] = word_ind.get(class_labels[i], unk_idx)
    
    # 統計標籤中<UNK>的使用情況
    unk_labels = sum(1 for label in class_labels if label == unk_idx)
    print(f"標籤中<UNK>的數量: {unk_labels} ({unk_labels/max(1,len(class_labels))*100:.2f}%)")
    
    # 轉換為張量
    tensor_predictors = []
    for i in range(len(pad_predictors)):
        tensor_predictors.append(torch.tensor(pad_predictors[i]))
    
    pad_predictors_tensor = torch.stack(tensor_predictors)
    class_labels_tensor = torch.tensor(class_labels)
    
    print(f"輸入張量形狀: {pad_predictors_tensor.shape}")
    print(f"標籤張量形狀: {class_labels_tensor.shape}")
     
    return pad_predictors_tensor, class_labels_tensor

class LSTM(nn.Module):
    """LSTM模型類"""
    def __init__(self, num_embeddings, embedding_dim, padding_idx, hidden_size, Dropout_p, batch_size, num_layers=2):
        super(LSTM, self).__init__()
        
        # 基本參數設置
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.hidden_size = hidden_size
        self.dropout_p = Dropout_p
        self.batch_size = batch_size
        self.num_layers = num_layers
        
        # 詞嵌入層
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        
        # 嵌入層dropout
        self.embed_dropout = nn.Dropout(Dropout_p * 0.5)
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=Dropout_p if num_layers > 1 else 0
        )
        
        # Dropout層
        self.dropout = nn.Dropout(Dropout_p)
        
        # 全連接層
        self.fc = nn.Linear(hidden_size, num_embeddings)
        
    def init_hidden(self, batch_size):
        """初始化隱藏狀態"""
        state_h = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        state_c = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        return (state_h, state_c)
        
    def forward(self, input_sequence, state_h, state_c):
        """前向傳播"""
        # 應用詞嵌入
        embedded = self.embedding(input_sequence)
        
        # 應用嵌入dropout
        embedded = self.embed_dropout(embedded)
        
        # 應用LSTM層
        output, (state_h, state_c) = self.lstm(embedded, (state_h, state_c))
        
        # 應用Dropout
        dropped = self.dropout(output[:, -1, :])
        
        # 應用全連接層
        logits = self.fc(dropped)
         
        return logits, (state_h, state_c)
    
    def topk_sampling(self, logits, topk, temperature=1.0):
        """安全的topk採樣函數，避免索引錯誤及排除<UNK>"""
        # 應用溫度縮放
        logits = logits / temperature
        
        # 應用softmax獲取概率分布
        logits_softmax = F.softmax(logits, dim=1)
        
        # 獲取<UNK>的索引和padding的索引
        unk_idx = word_ind.get("<UNK>", -1)
        pad_idx = word_ind.get("", -1)
        
        # 查找有效索引的最大值
        max_valid_idx = max(ind_word.keys())
        
        # 獲取全部概率和索引
        values, indices = torch.sort(logits_softmax[0], descending=True)
        
        # 過濾掉無效索引、<UNK>和padding
        valid_values = []
        valid_indices = []
        for i, idx in enumerate(indices.tolist()):
            # 確保索引存在於字典中，且不是特殊標記
            if idx in ind_word and idx != unk_idx and idx != pad_idx:
                valid_values.append(values[i])
                valid_indices.append(idx)
        
        # 如果沒有有效選擇，則選擇第一個非特殊標記的詞
        if not valid_indices:
            for idx in range(1, min(100, max_valid_idx + 1)):
                if idx in ind_word and idx != unk_idx and idx != pad_idx:
                    return ind_word[idx]
            # 最壞情況下返回一個安全的詞
            return ""
        
        # 只保留前topk個有效選擇
        max_k = min(topk, len(valid_values))
        valid_values = valid_values[:max_k]
        valid_indices = valid_indices[:max_k]
        
        # 從topk中按概率加權採樣
        try:
            probs = torch.tensor(valid_values) / sum(valid_values)
            idx = torch.multinomial(probs, num_samples=1).item()
            sampled_idx = valid_indices[idx]
            return ind_word[sampled_idx]
        except Exception as e:
            print(f"採樣過程出錯: {e}")
            # 出錯時返回第一個有效詞
            if valid_indices:
                return ind_word[valid_indices[0]]
            return ""

def get_batch(pad_predictors, class_labels, batch_size):
    """批次數據生成器，處理所有數據"""
    for i in range(0, len(pad_predictors), batch_size):
        end_idx = min(i + batch_size, len(pad_predictors))
        yield pad_predictors[i:end_idx], class_labels[i:end_idx]

def train_model(train_predictors, train_labels, val_predictors=None, val_labels=None, 
                n_vocab=None, embedding_dim=150, padding_idx=0, hidden_size=256, 
                Dropout_p=0.3, batch_size=128, lr=0.001, num_layers=2):
    """訓練LSTM模型，改進版：防止過早早停"""
    # 創建checkpoint目錄
    os.makedirs("./checkpoints", exist_ok=True)
    
    # 創建日誌文件
    log_file = f"./training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def log_message(message):
        """記錄消息到控制台和日誌文件"""
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    # 創建LSTM模型實例
    model = LSTM(n_vocab, embedding_dim, padding_idx, hidden_size, Dropout_p, batch_size, num_layers=num_layers)
    
    # 記錄模型信息
    total_params = sum(p.numel() for p in model.parameters())
    log_message(f"模型參數總數: {total_params}")
    log_message(f"模型配置: 嵌入維度={embedding_dim}, 隱藏大小={hidden_size}, "
                f"Dropout={Dropout_p}, 批次大小={batch_size}, 層數={num_layers}")
    
    # 損失函數
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # 優化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 學習率調度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=3, verbose=True
    )
    
    # 訓練參數
    num_epochs = 50       # 增加最大訓練輪數
    best_loss = float('inf')
    patience = 15         # 增加早停耐心值
    min_epochs = 5        # 設置最小必須訓練的輪數
    patience_counter = 0
    best_epoch = 0
    
    # 準備驗證集（如果未提供）
    if val_predictors is None or val_labels is None:
        log_message("未提供驗證集，使用訓練集的15%作為驗證集")
        val_size = int(0.15 * len(train_predictors))
        indices = list(range(len(train_predictors)))
        random.shuffle(indices)
        
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        val_predictors = train_predictors[val_indices]
        val_labels = train_labels[val_indices]
        train_predictors = train_predictors[train_indices]
        train_labels = train_labels[train_indices]
    
    # 記錄數據集大小
    log_message(f"訓練集大小: {len(train_predictors)}")
    log_message(f"驗證集大小: {len(val_predictors)}")
    
    # 訓練開始時間
    start_time = time.time()
    
    # 訓練循環
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()  # 設置為訓練模式
        
        total_train_loss = 0
        batch_count = 0
        
        # 訓練階段 - 修復批次大小不匹配問題
        for x, y in get_batch(train_predictors, train_labels, batch_size):
            # 獲取當前批次的實際大小
            curr_batch_size = x.size(0)
            
            # 為當前批次初始化隱藏狀態
            state_h, state_c = model.init_hidden(curr_batch_size)
            
            # 前向傳播
            logits, (state_h, state_c) = model(x, state_h, state_c)
            
            # 計算損失
            loss = criterion(logits, y)
            loss_value = loss.item()
            total_train_loss += len(x) * loss_value
            batch_count += len(x)
            
            # 反向傳播
            model.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 參數更新
            optimizer.step()
        
        # 計算整體訓練損失
        avg_train_loss = total_train_loss / len(train_predictors)
        train_ppl = np.exp(avg_train_loss)
        
        # 驗證階段 - 同樣修復批次大小不匹配問題
        model.eval()  # 設置為評估模式
        total_val_loss = 0
        
        with torch.no_grad():  # 關閉梯度計算
            for x, y in get_batch(val_predictors, val_labels, batch_size):
                # 獲取當前批次的實際大小
                curr_batch_size = x.size(0)
                
                # 為當前批次初始化隱藏狀態
                val_state_h, val_state_c = model.init_hidden(curr_batch_size)
                
                # 前向傳播
                logits, (val_state_h, val_state_c) = model(x, val_state_h, val_state_c)
                
                # 計算驗證損失
                val_loss = criterion(logits, y)
                total_val_loss += len(x) * val_loss.item()
        
        # 計算整體驗證損失
        avg_val_loss = total_val_loss / len(val_predictors)
        val_ppl = np.exp(avg_val_loss)
        
        # 計算epoch耗時
        epoch_time = time.time() - epoch_start
        
        # 更新學習率
        scheduler.step(avg_val_loss)
        
        # 輸出當前epoch的訓練和驗證結果
        log_message(f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Train Loss: {avg_train_loss:.6f}, Train PPL: {train_ppl:.2f}, "
                    f"Val Loss: {avg_val_loss:.6f}, Val PPL: {val_ppl:.2f}, "
                    f"Time: {epoch_time:.2f}s, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 定期保存檢查點
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"./checkpoints/model_epoch_{epoch+1}.pt"
            torch.save(model, checkpoint_path)
            log_message(f"檢查點已保存至 {checkpoint_path}")
        
        # 生成一些文本示例
        if (epoch + 1) % 5 == 0 or epoch == 0:
            try:
                gen_text = generate(model, init='', max_chars=20, topk=5)
                log_message("\n生成文本示例:")
                log_message(gen_text)
                log_message("")
            except Exception as e:
                log_message(f"生成文本時出錯: {e}")
        
        # 改進版早停機制，確保至少訓練min_epochs輪
        if epoch + 1 < min_epochs:
            # 前min_epochs輪不觸發早停，只保存最佳模型
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(model, "./best_model.pt")
                patience_counter = 0
                log_message(f"模型改善! 儲存在epoch {epoch+1}, 驗證損失: {avg_val_loss:.6f}")
        else:
            # 達到最小輪數後才考慮早停
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                torch.save(model, "./best_model.pt")
                patience_counter = 0
                log_message(f"模型改善! 儲存在epoch {epoch+1}, 驗證損失: {avg_val_loss:.6f}")
            else:
                patience_counter += 1
                log_message(f"模型未改善，耐心度: {patience_counter}/{patience}")
                if patience_counter >= patience and epoch + 1 >= min_epochs:
                    log_message(f"早停觸發! 最佳模型在epoch {best_epoch}, 驗證損失: {best_loss:.6f}")
                    break
    
    # 訓練總耗時
    total_time = time.time() - start_time
    log_message(f"訓練完成! 總耗時: {total_time:.2f}秒, 最佳epoch: {best_epoch}")
    
    # 保存詞彙表，以便後續推理使用
    vocab_data = {
        'word_ind': word_ind,
        'ind_word': ind_word,
        'seq_len': seq_len
    }
    torch.save(vocab_data, "./vocab_data.pt")
    log_message(f"詞彙表已保存至 ./vocab_data.pt，包含 {len(word_ind)} 個詞")
    
    # 載入最佳模型
    try:
        best_model = torch.load("./best_model.pt")
        return best_model, best_loss
    except Exception as e:
        log_message(f"載入最佳模型失敗: {e}，返回當前模型")
        return model, best_loss

def evaluate(model_path, vocab_path="./vocab_data.pt"):
    """評估模型在測試集上的表現，使用訓練時保存的詞彙表"""
    try:
        # 嘗試載入保存的詞彙表
        if os.path.exists(vocab_path):
            vocab_data = torch.load(vocab_path)
            global word_ind, ind_word, seq_len
            word_ind = vocab_data['word_ind']
            ind_word = vocab_data['ind_word'] 
            seq_len = vocab_data['seq_len']
            print(f"使用已保存的詞彙表，包含 {len(word_ind)} 個詞")
        else:
            print("警告：找不到保存的詞彙表，將重新創建（可能導致詞彙不一致）")
        
        # 載入測試數據
        test_data = load_data("./test.txt")
        data = test_data[:]
        data = Clean_data(data)
        
        # 只有在沒有保存詞彙表時才重新創建
        if not os.path.exists(vocab_path):
            sentences, words = split_data(data, num_sentences=-1)
            pad_predictors, class_labels = Convert_data(sentences, words, seq_len)
        else:
            # 使用已保存的詞彙表處理測試數據
            sentences = sent_tokenize(data)
            
            sent_sequences = []
            for i in range(len(sentences)):
                words_in_sent = str.split(sentences[i], ' ')
                words_in_sent = [w for w in words_in_sent if w.strip()]
                
                processed_words = []
                for word in words_in_sent:
                    if word in word_ind:
                        processed_words.append(word)
                    else:
                        processed_words.append("<UNK>")
                
                if len(processed_words) <= 1:
                    continue
                    
                for j in range(1, len(processed_words)):
                    if j <= seq_len:
                        sent_sequences.append(processed_words[:j])
                    elif j > seq_len and j < len(processed_words):
                        sent_sequences.append(processed_words[j-seq_len:j])
                    elif j > len(processed_words)-seq_len:
                        sent_sequences.append(processed_words[j-seq_len:])
                        
            # 過濾過短序列
            sent_sequences = [seq for seq in sent_sequences if len(seq) > 1]
            
            # 分割為預測器和標籤
            predictors = []; class_labels = []
            for i in range(len(sent_sequences)):
                predictors.append(sent_sequences[i][:-1])
                class_labels.append(sent_sequences[i][-1])
            
            # 填充
            pad_predictors = []
            for i in range(len(predictors)):
                emptypad = [''] * (seq_len-len(predictors[i])-1)
                emptypad.extend(predictors[i])
                pad_predictors.append(emptypad)
            
            # 轉換為索引
            unk_idx = word_ind.get("<UNK>", 0)
            for i in range(len(pad_predictors)):
                for j in range(len(pad_predictors[i])):
                    word = pad_predictors[i][j]
                    pad_predictors[i][j] = word_ind.get(word, unk_idx)
                class_labels[i] = word_ind.get(class_labels[i], unk_idx)
            
            # 轉換為張量
            tensor_predictors = []
            for i in range(len(pad_predictors)):
                tensor_predictors.append(torch.tensor(pad_predictors[i]))
            
            pad_predictors = torch.stack(tensor_predictors)
            class_labels = torch.tensor(class_labels)
        
        print("測試集序列數: ", len(pad_predictors))
        
        # 載入保存的模型
        model = torch.load(model_path)
        
        # 評估
        with torch.no_grad():
            model.eval()
            batch_size = 128
            total_loss = 0
            
            for x, y in get_batch(pad_predictors, class_labels, batch_size):
                # 獲取當前批次的實際大小
                curr_batch_size = x.size(0)
                
                # 為當前批次初始化隱藏狀態
                state_h, state_c = model.init_hidden(curr_batch_size)
                
                logits, (state_h, state_c) = model(x, state_h, state_c)

                # 計算損失
                criterion = nn.CrossEntropyLoss()
                loss = criterion(logits, y)
                total_loss += len(x) * loss.item()
                
            total_loss /= len(pad_predictors)
            
            return total_loss, np.exp(total_loss)
    except Exception as e:
        print(f"評估時出錯: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), float('inf')

def generate(model, init, max_chars=20, topk=5, temperature=0.8, diversity_factor=0.2):
    """改進的生成函數，增加多樣性並避免重複生成"""
    sentence = init
    current_chars = len(sentence.replace(" ", ""))  # 計算當前漢字數（不計空格）
    generated_words = set()  # 記錄已生成的詞，避免重複
    max_attempts = 40  # 防止無限循環
    
    try:
        # 循環直到生成足夠的字符
        for _ in range(max_attempts):
            if current_chars >= max_chars:
                break
                
            # 設置為評估模式
            model.eval()
            
            # 處理輸入序列
            input_indices = []
            for word in str.split(sentence, " ") if sentence else [""]:
                # 確保word存在於詞典中
                if word in word_ind:
                    input_indices.append(word_ind[word])
                else:
                    # 若詞不存在，使用UNK標記
                    input_indices.append(word_ind.get("<UNK>", 0))
                    
            # 調整序列長度
            if len(input_indices) < seq_len-1:
                input_tensor = [0] * (seq_len-len(input_indices)-1)
                input_tensor.extend(input_indices)
            else:
                input_tensor = input_indices[-seq_len+1:]
                
            # 轉換為tensor
            input_tensor = torch.tensor([input_tensor])
            
            # 初始化隱藏狀態 - 使用批次大小1
            state_h, state_c = model.init_hidden(1)
            
            # 前向傳播
            with torch.no_grad():
                out, (state_h, state_c) = model(input_tensor, state_h, state_c)
            
            # 動態調整溫度以增加多樣性
            current_temp = temperature * (1 + len(generated_words) * diversity_factor)
            
            # 嘗試生成不重複的詞
            attempt = 0
            max_word_attempts = 3
            while attempt < max_word_attempts:
                word = model.topk_sampling(out, topk, current_temp)
                
                # 如果詞已經使用過，增加溫度再試一次
                if word in generated_words and word:
                    attempt += 1
                    current_temp += 0.2  # 增加溫度促進多樣性
                    continue
                break
            
            # 如果生成了有效的詞
            if word and word != "" and (not sentence or word != str.split(sentence, ' ')[-1]):
                # 添加生成的詞到句子中
                sentence = sentence + " " + word if sentence else word
                # 記錄已生成的詞
                generated_words.add(word)
                # 更新字符計數
                current_chars = len(sentence.replace(" ", ""))
            else:
                # 避免無限循環
                break
    
    except Exception as e:
        print(f"生成文本時出錯: {e}")
        traceback.print_exc()
    
    # 後處理：去除多餘空格
    sentence = ' '.join(sentence.split())
    
    return sentence

def main():
    """主函數：數據處理、模型訓練與評估"""
    try:
        # 創建輸出目錄
        os.makedirs("./outputs", exist_ok=True)
        os.makedirs("./checkpoints", exist_ok=True)
        
        # 設置隨機種子
        random_seed = 42
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 加載訓練數據
        print("開始載入訓練數據...")
        train = load_data("./train.txt")
        data = train[:]
        data = Clean_data(data)
        
        print("清理數據完成，開始分割數據...")
        sentences, words = split_data(data, num_sentences=-1)
        
        print("分割數據完成，開始轉換數據...")
        pad_predictors, class_labels = Convert_data(sentences, words, seq_len)
        
        print(f"總序列數: {len(pad_predictors)}")
        print(f"詞彙表大小: {len(words)}")
        
        # 拆分訓練集和驗證集
        print("拆分訓練集和驗證集...")
        indices = list(range(len(pad_predictors)))
        random.shuffle(indices)
        train_size = int(0.85 * len(pad_predictors))
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        train_predictors = pad_predictors[train_indices]
        train_labels = class_labels[train_indices]
        val_predictors = pad_predictors[val_indices]
        val_labels = class_labels[val_indices]
        
        print(f"訓練集大小: {len(train_predictors)}")
        print(f"驗證集大小: {len(val_predictors)}")
        
        # 訓練模型
        print("\n開始訓練模型...")
        model, best_loss = train_model(
            train_predictors=train_predictors, 
            train_labels=train_labels, 
            val_predictors=val_predictors, 
            val_labels=val_labels,
            n_vocab=len(words), 
            embedding_dim=150,
            padding_idx=0, 
            hidden_size=256,
            Dropout_p=0.3,
            batch_size=128,
            lr=0.001,
            num_layers=2
        )
        
        # 保存最終模型
        torch.save(model, "./PTT_Model_final.pt")
        print("模型已保存至 ./PTT_Model_final.pt")
        
        # 評估訓練集損失 - 修復批次大小不匹配問題
        print("\n評估訓練集損失...")
        model.eval()
        train_loss = 0
        batch_size = 128
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for x, y in get_batch(train_predictors, train_labels, batch_size):
                # 獲取當前批次的實際大小
                curr_batch_size = x.size(0)
                
                # 為當前批次初始化隱藏狀態
                state_h, state_c = model.init_hidden(curr_batch_size)
                
                logits, (state_h, state_c) = model(x, state_h, state_c)
                loss = criterion(logits, y)
                train_loss += len(x) * loss.item()
        
        train_loss /= len(train_predictors)
        train_ppl = np.exp(train_loss)
        
        print(f"Loss on train data: {train_loss}")
        print(f"Perplexity on train data: {train_ppl}")
        print(f"Number of input sequences: {len(pad_predictors)}")
        
        # 評估測試集
        print("\n評估測試集...")
        test_loss, test_perplexity = evaluate("./best_model.pt")
        print(f"Loss on test data: {test_loss}")
        print(f"Perplexity on test data: {test_perplexity}")
        
        # 生成示例文本
        print("\n生成示例文本:")
        for i in range(5):
            try:
                print(f"\n示例 {i+1}:")
                # 使用不同的溫度和topk值以獲得多樣性
                temperature = 0.7 + i * 0.1  # 0.7, 0.8, 0.9, 1.0, 1.1
                k = 5 + i * 3              # 5, 8, 11, 14, 17
                generated_text = generate(model, init='', max_chars=20, topk=k, temperature=temperature)
                print(generated_text)
                
                # 保存生成的文本
                with open(f"./outputs/sample_{i+1}.txt", "w", encoding="utf-8") as f:
                    f.write(generated_text)
                    
            except Exception as e:
                print(f"生成示例 {i+1} 時出錯: {e}")
                traceback.print_exc()
        
        print("\n程序執行完成!")
        return best_loss
        
    except Exception as e:
        print(f"主函數執行錯誤: {e}")
        import traceback
        traceback.print_exc()
        return float('inf')

if __name__ == "__main__":
    import traceback
    try:
        # 記錄開始時間
        start_time = time.time()
        
        # 設置序列長度
        seq_len = 16
        
        # 執行主函數
        loss = main()
        
        # 計算總執行時間
        total_time = time.time() - start_time
        print(f"\n總執行時間: {total_time:.2f} 秒")
    except Exception as e:
        print(f"程序執行過程中發生錯誤: {e}")
        traceback.print_exc()