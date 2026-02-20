# Path Signatures for NLP — Pipeline 梳理

## 一、项目概述

本项目用 **Path Signatures + Transformers**，基于 Project Gutenberg 语料做**作者异常检测**：判断文本片段是否出自训练集中未见过的作者。核心思路是把文本序列当作路径，用 path signature 作为特征，结合 RoBERTa 的 stream embeddings。

---

## 二、整体数据流

```
raw (.txt) → text (.txt 去 header) → tokens / counts (可选)
                    ↓
            metadata.csv (书目元数据)
                    ↓
    compute_embeddings (RoBERTa) → embeddings zarr + encoded_input zarr
                    ↓
    df 筛选 (英文 + 有 embedding) + 划分 normal/impostor
                    ↓
    token mask (Top-K 高频 token) + 降维 (UMAP 或 Random Projection)
                    ↓
    embeddings_projected (每书若干 path)
                    ↓
    Path Signature (truncation level 1–4)
                    ↓
    Isolation Forest 异常检测 + KS 检验 → ROCAUC
```

---

## 三、Stage 1: 数据获取与预处理（gutenberg）

### 3.1 数据来源

- **Standardized Project Gutenberg Corpus (SPGC)**
- 脚本：`gutenberg/get_data.py`、`process_data.py`
- 数据：https://github.com/pgcorpus/gutenberg

### 3.2 目录与格式

| 目录 | 内容 | 格式 | 说明 |
|------|------|------|------|
| `metadata/` | 书目元数据 | `metadata.csv` | id, title, author, language, downloads, subjects, type |
| `data/raw/` | 原始文本 | `PG{id}_raw.txt` | 从 PG 下载的原始 UTF-8 |
| `data/text/` | 清洗后文本 | `PG{id}_text.txt` | 去掉 header/legal notice |
| `data/tokens/` | 分词 | `PG{id}_tokens.txt` | 每行一个 token |
| `data/counts/` | 词频 | `PG{id}_counts.txt` | 类型级词频 |

### 3.3 处理流程

1. `get_data.py`：rsync 下载 UTF-8 书籍到 `.mirror`，再复制到 `raw/`
2. `process_data.py`：对每个 raw 文件执行：
   - `strip_headers`：去 header
   - `tokenize_text`：分词
   - 输出到 `text/`、`tokens/`、`counts/`

---

## 四、Stage 2: Metadata 读取与筛选

### 4.1 读取

```python
df = pd.read_csv(Path(DATASET_PATH, 'metadata', 'metadata.csv'))
df.set_index('id', inplace=True, drop=False)
```

- **shape**：约 (74743, 9)
- **列**：id, title, author, authoryearofbirth, authoryearofdeath, language, downloads, subjects, type

### 4.2 筛选条件

1. `df = df[df['author'].notnull()]`
2. `df = df[df['language'] == "['en']"]`：仅英文
3. 与 `successful_embeddings` 合并，只保留成功计算 embedding 的书

---

## 五、Stage 3: Compute Stream Embeddings

### 5.1 概念与参数

- 每本书取中间 10 个 512-token 的 chunk
- 每个 chunk 通过 RoBERTa 得到 512×1024 的 stream embedding
- 参数：
  - `CHUNK_SIZE = 512`
  - `N_CHUNKS = 10`
  - `BERT_DIMENSIONALITY = 1024`
- 前置条件：文本 token 数 ≥ 5120

### 5.2 `compute_embeddings` 逻辑

1. 读取 `PG{id}_text.txt`，合并为一行，去多余空格
2. `tokenizer(text)` 得到 token 序列
3. 取中间 5120 个 token：`strt_idx = length//2 - 2560`, `end_idx = length//2 + 2560`
4. 每 512 个 token 喂进 RoBERTa，得到 `last_hidden_state`
5. 输出：
   - `embeddings`：`(10, 512, 1024)` float32
   - `all_encoded_input`：`(10, 512)` int64

### 5.3 输出存储（zarr）

| 文件 | shape | dtype | 说明 |
|------|-------|-------|------|
| `embeddings` | (N, 10, 512, 1024) | float32 | N 本书的 stream embeddings |
| `encoded_input` | (N, 10, 512) | int64 | token IDs |
| `successful_embeddings.csv` | — | — | book_id → zarr 索引 |

### 5.4 数据与 df 的对应

```python
successful_embeddings = pd.read_csv(..., index_col=0, names=['embedding_index'])
df = df.merge(successful_embeddings, left_index=True, right_index=True)
```

- df 新增 `embedding_index`：指向 zarr 行

---

## 六、Stage 4: 数据集划分（异常检测设定）

### 4.1 任务

- 正常：训练集中出现过的作者
- 异常：Margaret Oliphant 作为 "impostor"

### 4.2 划分规则

```python
df_impostor = df[df['author'] == 'Oliphant, Mrs. (Margaret)']

df_normal = df[df['author'] != 'Oliphant, Mrs. (Margaret)']
df_normal_authors = df_normal.groupby('author').filter(lambda g: len(g) >= 10)
df_normal_eval = df_normal_with_multiple_per_author.groupby('author').sample(n=1)
df_normal_train = df_normal[df_normal['id'] not in df_normal_eval['id']]
```

- 正常作者至少 10 本书；eval 中每作者抽 1 本；其余为 train

---

## 七、Stage 5: Token 筛选与降维

### 5.1 四种 Dataset 配置

| 名称 | Top-K tokens | 降维方式 | 输出维度 |
|------|--------------|----------|----------|
| dataset0 | 250 | UMAP | 4 |
| dataset1 | 100 | UMAP | 2 |
| dataset2 | 100 | Random Projection | 2 |
| dataset3 | 250 | Random Projection | 4 |

### 5.2 `Dataset` 类

- 从 zarr 加载 token 与 embedding：
  - `token_ids = ENCODED_INPUT[embedding_index]`：`(10, 512)` masked
  - `embeddings = EMBEDDINGS[embedding_index]`：`(10, 512, 1024)`
- 按 Top-K 做 mask：只保留高频 token 对应位置
- `compute_mean()`：按 token 频率加权得到全局均值，用于去中心化

### 5.3 降维

**UMAP（dataset0 / dataset1）：**

1. 从训练集取 2000 个 chunk (1000 train + 1000 val)
2. 用 token 逆频率做采样权重
3. 在 1024 维 embedding 上 fit UMAP
4. 用 MLP（1024→1024→n_components）拟合 UMAP 映射，用于新数据
5. `project_embeddings(lambda em: ann.predict(em))`

**Random Projection（dataset2 / dataset3）：**

- 生成随机矩阵：`(1024, 2)` 或 `(1024, 4)`
- `projected = (em - projection_mean_offset) @ projection_matrix`

### 5.4 投影后 shape

- `embeddings_projected_normal_train`：list of list，每个元素 shape `(L, d)`
  - L：该 chunk 中 Top-K 内的观测数
  - d：2 或 4

---

## 八、Stage 6: Path Signature 与异常检测

### 6.1 Path Signature

```python
def compute_signature(path, truncation_level=2, cumsum_transform=True):
    path = np.vstack((np.zeros((2, path.shape[1])), path))
    if cumsum_transform:
        path = np.cumsum(path, axis=0)
    signature = iisignature.sig(path, truncation_level)
    return signature
```

- 输入：`path` shape `(L, d)`，d=2 或 4
- 先加零起点，再做 cumsum（可选）
- 输出：level-`truncation_level` 的 signature，维数由 iisignature 决定（level 4 ≈ 340 维）

### 6.2 异常检测流程 (`compute_anomaly_scores`)

1. 从 normal paths 中随机抽取 train / test
2. 对每个 path 计算 signature
3. 用 Isolation Forest 在 normal signatures 上 fit
4. 用 normal test 的 scores 做校准分布 `F1`
5. 对 impostor 的 scores 得到 `F2`
6. KS 检验比较 `F1`、`F2`，p-value 作为异常分数（越高越像异常）
7. 用 ROC 曲线评估，得到 ROCAUC

### 6.3 多 chunk 聚合

- 一本书有多条 path（多条 chunk），对应多个 signature
- 对一本书的所有 signature 做 score，再在**整本书**层面做 KS 检验
- 因此异常分数是针对"整本书"的

---

## 九、Shape 与数据流速查

| 阶段 | 数据 | Shape / 格式 |
|------|------|--------------|
| Raw text | PG{id}_text.txt | 纯文本，变长 |
| Tokenized | 中间变量 | 1D 序列，长度 ≥ 5120 |
| 中心 5120 tokens | 中间变量 | (5120,) int64 |
| Chunks | 中间变量 | (10, 512) int64 |
| RoBERTa output | 每 chunk | (512, 1024) float32 |
| embeddings (单本书) | zarr 行 | (10, 512, 1024) |
| encoded_input (单本书) | zarr 行 | (10, 512) int64 |
| Token mask 后 | masked array | (10, 512) 部分有效 |
| 降维后单 path | path 数组 | (L, 2) 或 (L, 4) |
| Path signature | 单 path | 1D，维度由 level 决定 |
| 异常分数 | scalar | p-value |

---

## 十、关键文件路径

| 用途 | 路径 |
|------|------|
| 数据根目录 | `DATASET_PATH` = `gutenberg/` |
| Embeddings 输出 | `EMBEDDINGS_PATH` = `/ssd/KarenHE/gutenberg_embeddings` |
| metadata | `gutenberg/metadata/metadata.csv` |
| text | `gutenberg/data/text/PG{id}_text.txt` |
| embeddings zarr | `{EMBEDDINGS_PATH}/embeddings` |
| encoded_input zarr | `{EMBEDDINGS_PATH}/encoded_input` |
| successful_embeddings | `{EMBEDDINGS_PATH}/successful_embeddings.csv` |
| embedding_datasets | `gutenberg/data/embedding_datasets.pkl`（Dataset 实例） |

---

## 十一、依赖与模型

- **模型**：`roberta-large`（transformers）
- **降维**：UMAP、sklearn MLPRegressor、Random Projection
- **异常检测**：sklearn IsolationForest
- **Path Signature**：`iisignature`
