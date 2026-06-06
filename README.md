# Motion Capture — ポーズ操作で動く3Dロボット

Webカメラの前での**全身の動き**をリアルタイムに3Dロボットへ反映するモーションキャプチャ・パイプラインです。
腕・脚・体幹・頭は実際のポーズに**忠実にミラー**され、移動だけは**足踏み（前進）と上体の向き（旋回）**で
意図的に操作できます。

```
[カメラ] → [YOLO11-pose 推論 (Python)] → UDP → [Unity ロボット]
                                                  ├─ 全身ミラー（ボーンへ姿勢を反映）
                                                  └─ ロコモーション（足踏み=前進 / 向き=旋回 / 両手上げ=ジャンプ）
```

---

## リポジトリ構成

モノレポ。2つのプロジェクトがUDPで連携して1つのシステムを構成します。

| ディレクトリ | 内容 |
|---|---|
| `pose_estimation/` | Python製のポーズ推定アプリ（YOLO11-pose）。カメラ→キーポイント／特徴→UDP送信 |
| `MothionCapture/` | Unityプロジェクト。UDP受信→全身ミラー＋ロコモーション |
| `doc/` | 設計ドキュメント（`implementation-plan.md` / `usage-guide.md`） |

---

## 必要環境

- **Python** 3.12 ＋ [uv](https://docs.astral.sh/uv/)
- **Unity** 6000.4.7f1 (LTS)
- **GPU**: NVIDIA GPU ＋ CUDA 12.6（CPUでも動作するが非推奨）
- **Webカメラ**: 全身（頭〜足首）が映る距離に設置できること

---

## セットアップ

### 1. Python（pose_estimation）

```powershell
cd pose_estimation
uv sync
```

初回起動時に YOLO11n-pose モデル（約6MB）が自動ダウンロードされます。

### 2. Unity（MothionCapture）

1. Unity Editor で `MothionCapture` プロジェクトを開く（スクリプトが自動コンパイル）
2. `Assets/Prefabs/PlayerRobot.prefab` を開き、`Robot` GameObject に以下を `Add Component`:
   - **PoseInputReceiver**（UDP受信＋ロコモーション）
   - **PoseAvatarDriver**（全身ボーンミラー）
3. 参照フィールドは空でOK（同一GameObjectから自動取得）。保存。

> 詳細な手順は [`doc/usage-guide.md`](doc/usage-guide.md) を参照してください。

---

## 実行

**Python を先に起動 → Unity を Play** の順で起動します。

```powershell
# 1) Python（全身がカメラに映る位置に立つ）
cd pose_estimation
uv run python main.py
```

```
# 2) Unity Editor で Play ▶
#    コンソールに次が出れば接続成功:
#    [PoseInputReceiver] Listening on UDP port 5005
```

---

## 操作方法

### 全身ミラー（常時）

腕・脚・体幹・頭が常にあなたのポーズをミラーします。YOLOは2D推定のため、
**カメラに正対するほど忠実**になります（奥行き・ひねりは平面に潰れるのが原理的限界）。

### 移動（意図的な操作）

| ジェスチャー | 動作 |
|---|---|
| その場で足踏み（左右の足を交互に上下） | 前進（速いほど速い） |
| 上体を左右にひねる／傾ける | その方向へ旋回 |
| 両手首を鼻より高く上げる | ジャンプ |

### デバッグキー（Pythonウィンドウ）

| キー | 動作 |
|---|---|
| `q` | 終了 |
| `p` | 全キーポイント座標を出力 |
| `g` | ロコモーション特徴（forward/turn/jump/confidence）を出力 |

---

## アーキテクチャ概要

二層構成。**同一のUDPパケット（`127.0.0.1:5005`, JSON v2）**を共有します。

- **ミラー層** — `PoseAvatarDriver` が `LateUpdate`（Animator評価後）で2Dキーポイントを
  Humanoidボーンへリターゲット（平面・信頼度ゲート・Slerp平滑化）。
- **ロコモーション層** — `gesture_mapper` が足踏み/向き/ジャンプを特徴量化し、
  `ThirdPersonController` のポーズ移動モードがルートを駆動。

### パケット形式（v2）

```json
{"v":2, "kp":[x0,y0,...x16,y16], "kp_conf":[...17],
 "forward":0..1, "turn":-1..1, "jump":false, "confidence":0..1}
```

`kp` はフラットなfloat配列（Unityの `JsonUtility` がジャグ配列を扱えないため）。
信頼度 < 0.4 または0.5秒パケットが途切れると、移動は停止しミラーは最後の姿勢を保持します。

> 設計の詳細は [`doc/implementation-plan.md`](doc/implementation-plan.md) を参照。

---

## チューニング

| 対象 | 場所 | 主なパラメータ |
|---|---|---|
| 前進（足踏み）感度 | `pose_estimation/gesture_mapper.py` | `STEP_MIN_AMP`, `STEP_FULL_CROSS` |
| 旋回感度 | `pose_estimation/gesture_mapper.py` | `TURN_DEADZONE`, `TURN_SCALE` |
| ジャンプ誤検知 | `pose_estimation/gesture_mapper.py` | `JUMP_HOLD_FRAMES` |
| ミラー左右・追従 | Unity `PoseAvatarDriver` | `mirror`, `swapSides`, `responsiveness` |
| 旋回の速さ | Unity `ThirdPersonController` | `TurnSpeed` |

---

## 主な技術スタック

- **Python**: PyTorch (CUDA 12.6) ＋ Ultralytics YOLO11-pose ＋ OpenCV ＋ NumPy
- **Unity**: URP 17.4 ／ New Input System 1.19 ／ Cinemachine 3.1 ／ CharacterController ベース

---

## トラブルシューティング

- `Cannot bind port 5005` … `PoseInputReceiver` の重複、もしくはポート使用中。`UDP_PORT` と `Udp Port` を変更。
- ロボットが動かない … Pythonが起動中か、`confidence` が0.4以上か、コンポーネントが `Robot` に付いているか確認。
- 推論が遅い … コンソールが `Using device: CPU` ならGPU/CUDAのセットアップを確認。

詳細は [`doc/usage-guide.md`](doc/usage-guide.md) を参照してください。
