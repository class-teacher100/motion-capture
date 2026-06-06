# pose-estimation

YOLO11-pose を使ったリアルタイム姿勢推定アプリ。カメラ映像から複数人の姿勢を同時に検出し、17点のキーポイント座標をリアルタイムで取得する。

## 概要

- **モデル**: YOLO11n-pose（COCOキーポイント 17点）
- **対応人数**: 複数人同時検出
- **高速推論**: GPU（CUDA）対応、RTX 4060 で 60〜90 FPS
- **出力情報**: 各キーポイントの (x, y) ピクセル座標 + 信頼度スコア

### 検出キーポイント（17点）

| ID | 部位 | ID | 部位 |
|---|---|---|---|
| 0 | 鼻 | 9 | 左手首 |
| 1 | 左目 | 10 | 右手首 |
| 2 | 右目 | 11 | 左腰 |
| 3 | 左耳 | 12 | 右腰 |
| 4 | 右耳 | 13 | 左膝 |
| 5 | 左肩 | 14 | 右膝 |
| 6 | 右肩 | 15 | 左足首 |
| 7 | 左肘 | 16 | 右足首 |
| 8 | 右肘 | | |

## 環境要件

- OS: Windows 10/11 (64-bit)
- Python: 3.12
- GPU: NVIDIA GPU（CUDA 12.6 以上推奨）※ CPU でも動作可
- カメラ: USB カメラまたは内蔵カメラ

## 環境構築

### 1. uv のインストール

PowerShell を開いて以下を実行する。

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

インストール後、新しいターミナルを開いて動作確認。

```powershell
uv --version
```

### 2. リポジトリのクローン

```powershell
git clone <repository-url>
cd pose_estimation
```

### 3. 依存パッケージのインストール

Python 3.12 の自動セットアップと全パッケージのインストールを一括で行う。

```powershell
uv sync
```

> **GPU 版 PyTorch について**  
> `pyproject.toml` に PyTorch 公式の CUDA 12.6 インデックスを設定済みのため、`uv sync` だけで GPU 対応版が自動的にインストールされる。

## 実行

```powershell
uv run python main.py
```

初回起動時に YOLO11n-pose のモデルファイル（約 6 MB）が自動ダウンロードされる。

### 操作方法

| キー | 動作 |
|---|---|
| `q` | 終了 |
| `p` | その瞬間の全キーポイント座標を標準出力に表示 |

### 画面表示

- **緑の点**: 検出されたキーポイント
- **オレンジの線**: スケルトン（骨格）
- **左上**: FPS・検出人数・推論デバイス

## カスタマイズ

`main.py` 冒頭の定数で動作を調整できる。

```python
CONF_THRESHOLD = 0.5      # 人物検出の信頼度しきい値（0〜1）
KP_CONF_THRESHOLD = 0.5   # キーポイントの信頼度しきい値（0〜1）
MODEL_NAME = "yolo11n-pose.pt"  # 使用モデル
CAMERA_INDEX = 0          # カメラ番号（複数カメラがある場合は 1, 2, ... に変更）
```

### モデルの選択

精度と速度のトレードオフに応じてモデルを選択する。

| モデル | サイズ | 速度 | 精度 |
|---|---|---|---|
| `yolo11n-pose.pt` | 6 MB | 最速 | 標準 |
| `yolo11s-pose.pt` | 23 MB | 速い | 高い |
| `yolo11m-pose.pt` | 89 MB | 中程度 | より高い |
| `yolo11l-pose.pt` | 197 MB | 遅い | 高精度 |
