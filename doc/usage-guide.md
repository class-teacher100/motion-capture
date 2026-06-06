# 使い方ガイド：ポーズでロボットを全身ミラー操作する

カメラの前での全身の動きを Unity のロボットに**忠実にミラー**しつつ、移動だけは
**足踏み（前進）と体の向き（旋回）**で意図的に操作できます。

## 必要なもの

- NVIDIA GPU（CUDA 12.6 対応）※ CPU でも動作するが推奨しない
- Webカメラ（全身＝頭から足首まで映る距離が必要）
- Unity 6000.4.7f1
- Python 3.12 + uv

---

## セットアップ（初回のみ）

### 1. Python 依存パッケージをインストール

```powershell
cd F:\Users\cre\sut\pose_estimation
uv sync
```

初回起動時に YOLO11n-pose モデル（約 6 MB）が自動ダウンロードされます。

### 2. Unity でコンポーネントを設定

ポーズ操作には 2 つのコンポーネントを **`PlayerRobot` プレハブ内の `Robot` GameObject**
（`Animator` / `ThirdPersonController` / `StarterAssetsInputs` が付いているオブジェクト）に追加します。

1. Unity Editor で `MothionCapture` プロジェクトを開く（スクリプトが再コンパイルされる）
2. `Assets/Prefabs/PlayerRobot.prefab` をダブルクリックして Prefab 編集モードへ
3. `Robot` GameObject を選択し、`Add Component` →
   - **PoseInputReceiver**（受信＋ロコモーション）
   - **PoseAvatarDriver**（全身ボーンミラー）
4. 参照フィールドは空のままで OK（同一 GameObject から自動取得）。明示する場合は
   - `PoseAvatarDriver.Receiver` ← 同じ `Robot`（PoseInputReceiver）
   - `PoseAvatarDriver.Animator` ← `Robot` の Animator
5. `Ctrl+S` で保存

> **重要**: アセット本体と `.meta` ファイルは常にセットで Git 管理してください。
> 新規 `PoseAvatarDriver.cs.meta` と `PlayerRobot.prefab` の変更を忘れずにコミット。

---

## 起動手順

### ステップ 1: Python を起動

```powershell
cd F:\Users\cre\sut\pose_estimation
uv run python main.py
```

カメラ映像と骨格描画のウィンドウが開きます。**全身（頭〜足首）**がカメラに映っていることを確認してください。

### ステップ 2: Unity を Play モードにする

Unity Editor で **Play ボタン**（▶）を押します。コンソールに以下が出れば接続成功です:

```
[PoseInputReceiver] Listening on UDP port 5005
```

### ステップ 3: 操作する

カメラの前で体を動かすと、ロボットの腕・脚・体幹・頭がリアルタイムにミラーされます。
移動は下記のジェスチャーで行います。

---

## 操作方法

### 全身ミラー（常時）

腕・脚・体幹・頭は**常にあなたのポーズを忠実にミラー**します（歩行アニメは再生されません）。
YOLO は 2D 推定のため、**カメラに正対するほど忠実**になります。奥行き方向の動き（前後の腕振り等）や
体のひねりは平面に潰れる点が原理的な限界です。

### 移動（意図的な操作）

| ジェスチャー | キャラクター操作 |
|---|---|
| その場で**足踏み**（左右の足を交互に上下） | 前進（足踏みが速いほど速い） |
| **上体を左右にひねる／傾ける** | その方向へ旋回 |
| **両手首を鼻より高く上げる** | ジャンプ |

> 足踏みをやめると前進が止まり、上体を正面に戻すと旋回が止まります。
> 移動操作は「体の傾き方式」ではなく足踏み＋向きにしているため、全身ミラーと干渉しにくくなっています。

---

## コントロールキー（Python ウィンドウにフォーカスした状態で）

| キー | 動作 |
|-----|------|
| `q` | 終了 |
| `p` | 現在フレームの全キーポイント座標をコンソール出力 |
| `g` | 現在フレームのロコモーション特徴をコンソール出力 |

### `[g]` の出力例

```
Features (person 1): {'forward': 0.842, 'turn': -0.31, 'jump': False, 'confidence': 0.87}
```

---

## 反応がうまくいかない場合

### 移動感度の調整（`pose_estimation/gesture_mapper.py`）

```python
# 足踏み（前進）
STEP_MIN_AMP   = 0.15   # 大きくすると足を高く上げないと前進しない
STEP_FULL_CROSS = 3.0   # 小さくすると軽い足踏みで最大速度に達する
STEP_WINDOW_S  = 1.0    # ケイデンス測定の時間窓（秒）

# 旋回（体の向き）
TURN_DEADZONE = 0.08    # 大きくすると旋回の誤発火が減る
TURN_SCALE    = 0.35    # 小さくすると小さなひねりで最大旋回になる
```

### ミラー／追従の調整（Unity `PoseAvatarDriver`）

| 症状 | 調整 |
|---|---|
| 左右が逆 | **Mirror** をオフ、または **Swap Sides** をオン |
| 手足がプルプル震える | **Responsiveness** を下げる（例 8〜10） |
| 追従が遅い | **Responsiveness** を上げる（例 20〜25） |

### 旋回の速さ（Unity `ThirdPersonController`）

- **Turn Speed**（既定 120 deg/秒）を増減して旋回の速さを調整。

### カメラ位置の推奨設定

- カメラから **1.5〜2.5 m** 程度離れる
- カメラの高さは **腰〜胸の高さ**
- 全身（頭から足首まで）がフレームに収まることを確認（足首が映らないと足踏み検出が効きません）

### ジャンプが誤検知する

両手首が同時に鼻より上にある状態が 3 フレーム以上続くとジャンプします。誤発火する場合は
`JUMP_HOLD_FRAMES` を増やしてください:

```python
JUMP_HOLD_FRAMES = 5  # 3→5 に増やすと誤検知が減る
```

### キャラクターが止まらない／ミラーが固まる

カメラから外れる等で信頼度が下がると、ロコモーションは停止し、ミラーは**最後の良い姿勢を保持**します
（0.5 秒のタイムアウト後）。完全に終了するには Python ウィンドウで `[q]` を押してください。

---

## Inspector のオプション

### PoseInputReceiver（受信＋ロコモーション）

| フィールド | デフォルト | 説明 |
|-----------|-----------|------|
| `Pose Control Enabled` | true | ポーズ操作の有効化（OFF でキーボード操作に戻る） |
| `Udp Port` | 5005 | 受信ポート番号（Python 側と合わせる） |
| `Min Confidence` | 0.4 | これ未満の信頼度のパケットは無視 |
| `Packet Timeout Seconds` | 0.5 | この秒数パケットがこないとゼロ入力＋ミラー保持 |

### PoseAvatarDriver（全身ミラー）

| フィールド | デフォルト | 説明 |
|-----------|-----------|------|
| `Mirror` | true | 鏡像の操作感（左右反転） |
| `Swap Sides` | false | 左右のキーポイント対応を入れ替え |
| `Kp Min Conf` | 0.4 | これ未満のキーポイントは該当ボーンの更新を保留 |
| `Responsiveness` | 15 | 大きいほど追従が速くジッタ増、小さいほど滑らか |

### ThirdPersonController（移動）

| フィールド | デフォルト | 説明 |
|-----------|-----------|------|
| `Pose Locomotion` | false | 実行時に PoseInputReceiver が自動で true にする |
| `Turn Speed` | 120 | 旋回の速さ（deg/秒） |

> **ポートを変更する場合**: `pose_sender.py` の `UDP_PORT` と Inspector の `Udp Port` を同じ値にしてください。

---

## キーボード操作との関係

ポーズ操作モード中は、安定動作のため以下が自動で設定されます（`PoseInputReceiver.Start`）:

- `PlayerInput` を無効化（キーボード/ゲームパッドの上書きを防止）
- `ThirdPersonController.PoseLocomotion = true`（足踏み＋向きの移動に切替）
- `LockCameraPosition = true`（マウス無しのためカメラ固定）

キーボード操作に戻すには Inspector で **`Pose Control Enabled` を OFF** にして再生してください。

---

## トラブルシューティング

### `[PoseInputReceiver] Cannot bind port 5005` と表示される

ポート 5005 が他のプロセスに使用されています。シーン内に `PoseInputReceiver` が**重複していないか**確認し、
重複が無ければ `pose_sender.py` の `UDP_PORT` と Unity の `Udp Port` を別番号（例: 5006）に変更してください。

### ロボットが全く動かない

- Python が起動しているか、`[g]` で `confidence` が 0.4 以上になっているか確認
- Console にコンパイルエラーが出ていないか確認
- `PoseInputReceiver` と `PoseAvatarDriver` が同じ `Robot` に付いているか確認

### カメラが開かない

`main.py` の `CAMERA_INDEX = 0` を `1` や `2` に変更して別のカメラを試してください。

### 推論が遅い（FPS が低い）

GPU が使用されていない可能性があります。コンソールに `Using device: CPU` と表示されている場合、
CUDA 12.6 ドライバーと PyTorch の GPU 版が正しくインストールされているか確認してください。
