# 実装計画：ポーズ推定 → Unity ロボット 全身ミラー＋意図的ロコモーション

> **更新（v3 / 3D 化）**: 推論を YOLO11（2D）から **MediaPipe Pose（3D）** に変更。
> パケットは v3 になり、従来の 2D `kp`/`kp_conf` に加えて 3D ワールド座標 `kp3d[51]`
> （メートル・腰中心）を送信する。`PoseAvatarDriver` は `kp3d` で奥行きを含む真の 3D
> ボーン方向を計算し（`useDepth`/`depthScale`）、`kp3d` が無い場合は従来の 2D 平面パス
> にフォールバックする。`gesture_mapper.py` は 2D のままで変更なし。以下の本文中の
> 「YOLO11」「v2」「2D 平面」等の記述はこの上書きを反映した読み替えが必要。

## 概要

`pose_estimation`（Python/MediaPipe Pose）で取得した人体姿勢データ（2D＋3D）を
リアルタイムに `MothionCapture`（Unity）へ送信し、カメラの前での全身の動きで
3D ロボットを操作する。

設計は**二層構成**:

1. **ミラー層（忠実性）**: 17 キーポイントを Humanoid アバターのボーンへ常時リターゲットし、
   腕・脚・体幹・頭を忠実にミラーする。
2. **ロコモーション層（操作性）**: 移動だけは意図的に操作できる専用ジェスチャ
   （**足踏み＝前進 / 上体の向き＝旋回 / 両手上げ＝ジャンプ**）で制御する。

両層は**同一の UDP パケット**を共有し、Unity 側で別コンポーネントが処理する。

---

## アーキテクチャ

```
[カメラ]
  ↓
[YOLO11n-pose 推論]            ← GPU加速（CUDA 12.6）
  ↓ 17 キーポイント(2D)
[GestureMapper]                ← 足踏み→forward / 上体の向き→turn / 両手上げ→jump
  ↓                              ＋正規化キーポイントを併載
  │ UDP 127.0.0.1:5005（JSON v2、約400バイト/フレーム）
  ↓
[PoseInputReceiver.cs]         ← バックグラウンドスレッドで受信（単一の受信口）
  ├──→ [PoseAvatarDriver.cs]   ← LateUpdate で 17kp→ボーン回転（全身・平面・平滑化）
  │                               ＝ ミラー層
  └──→ [ThirdPersonController]  ← forward/turn/jump でルート移動（ポーズ移動モード）
                                  ＝ ロコモーション層
  ↓
[3D ロボット：全身ミラー＋意図的移動]
```

### 通信方式の選定

| 方式 | レイテンシ | 難易度 | 採否 |
|------|-----------|--------|------|
| **UDP ループバック** | 〜0.1 ms | 低 | **採用** |
| 名前付きパイプ | 〜0.5 ms | 中 | 非採用 |
| 共有メモリ | 〜0.05 ms | 高 | 非採用 |
| 仮想ゲームパッド | 〜5 ms | 非常に高 | 非採用 |

UDP を採用した理由: Python 標準ライブラリのみで実装可能、Unity も外部パッケージ不要、
パケットロスト時は直前の入力状態を維持するため安全。

---

## パケット形式（v2）

Python → Unity へ毎フレーム送信する JSON:

```json
{
  "v": 2,
  "kp":      [x0, y0, x1, y1, ... x16, y16],
  "kp_conf": [c0, c1, ... c16],
  "forward": 0.0,
  "turn":    0.0,
  "jump":    false,
  "confidence": 0.85
}
```

| フィールド | 型 | 値域 | 説明 |
|-----------|-----|------|------|
| `v` | int | 2 | スキーマバージョン |
| `kp` | float[34] | 0.0〜1.0 | 正規化キーポイント（x,y 交互、画像幅/高で除算） |
| `kp_conf` | float[17] | 0.0〜1.0 | 各キーポイントの信頼度 |
| `forward` | float | 0.0〜1.0 | 前進速度スケール（足踏みケイデンス） |
| `turn` | float | −1.0〜1.0 | 旋回率（左が負、上体の向き） |
| `jump` | bool | true/false | ジャンプトリガー |
| `confidence` | float | 0.0〜1.0 | ロコモーション用キーポイントの平均信頼度 |

- `kp` は**フラット配列**（`[[x,y],...]` ではない）。Unity の `JsonUtility` がジャグ配列を
  デシリアライズできないため、`float[]` に展開している。
- `confidence < 0.4` または 0.5 秒以上パケットが来ない場合、Unity 側はロコモーションをゼロ入力にし、
  ミラーは**最後の良い姿勢を保持**する。

---

## ジェスチャーマッピング（`gesture_mapper.py`）

座標は正規化（÷幅, ÷高）。スケールはすべて体幹高 `torso_h`（肩中点〜腰中点の鉛直距離）で割ることで、
カメラ距離・体格の差を自己補正する。

### 使用キーポイント（COCO 17点）

| インデックス | 部位 | 用途 |
|-------------|------|------|
| 0 | 鼻 | ジャンプ・首/頭ミラー |
| 5, 6 | 左右肩 | 旋回・体幹/腕ミラー |
| 7, 8 | 左右肘 | 腕ミラー |
| 9, 10 | 左右手首 | ジャンプ・腕ミラー |
| 11, 12 | 左右腰 | 旋回・脚/体幹ミラー |
| 13, 14 | 左右膝 | 足踏み（補助）・脚ミラー |
| 15, 16 | 左右足首 | 足踏み（主）・脚ミラー |

### 前進（forward）：足踏み検出

左右の足の鉛直位置の**交互振動（ケイデンス）**を検出する。足首を優先し、低信頼時は膝で代用。

```
signal = (right_foot_y - left_foot_y) / torso_h        # 足が交互に動くと符号が振動
時間窓 STEP_WINDOW_S(=1s) 内で:
  amplitude = max(signal) - min(signal)
  crossings = detrend(signal) の符号反転回数（STEP_HYSTERESIS でノイズ除去）
  forward   = (amplitude >= STEP_MIN_AMP) ? clamp(crossings/秒 / STEP_FULL_CROSS, 0, 1) : 0
足踏みが無ければ STEP_IDLE_DECAY で 0 に減衰
```

### 旋回（turn）：上体の向き

肩中点と腰中点の**水平オフセット**（上体のひねり/傾き）から旋回を求める。

```
offset = (shoulder_mid_x - hip_mid_x) / torso_h
turn   = (|offset| > TURN_DEADZONE) ? sign(offset) * clamp((|offset|-TURN_DEADZONE)/TURN_SCALE, 0, 1) : 0
```

> 体幹を直接ミラーするため、旧「体の傾き／横ずれ」方式は誤動作しやすく廃止。
> 足踏み＋向きはミラーと干渉しにくい。

### ジャンプ（jump）

両手首が鼻より上に 3 フレーム以上保持 → 発火。発火後 `JUMP_COOLDOWN_S`(0.55s) クールダウン
（`ThirdPersonController.JumpTimeout` と整合）。

```
jump = (kp[9].y < kp[0].y) AND (kp[10].y < kp[0].y)  → JUMP_HOLD_FRAMES 連続 AND クールダウン終了
```

### 調整可能な定数

| 定数 | デフォルト | 説明 |
|------|-----------|------|
| `KP_MIN_CONF` | 0.40 | 有効キーポイントの最低信頼度 |
| `EMA_ALPHA` | 0.35 | forward/turn の平滑化係数（小さいほど滑らか） |
| `STEP_WINDOW_S` | 1.0 | ケイデンス測定の時間窓（秒） |
| `STEP_MIN_AMP` | 0.15 | 足踏みと判定する最小振幅（体幹高単位） |
| `STEP_HYSTERESIS` | 0.03 | 微小な符号反転をノイズとして無視 |
| `STEP_FULL_CROSS` | 3.0 | 最大前進に対応する符号反転/秒 |
| `STEP_IDLE_DECAY` | 0.80 | 足踏み無しのときの forward 減衰率/フレーム |
| `TURN_DEADZONE` | 0.08 | 旋回のデッドゾーン |
| `TURN_SCALE` | 0.35 | 最大旋回に対応する水平オフセット量 |
| `JUMP_HOLD_FRAMES` | 3 | ジャンプ発火に必要なフレーム数 |
| `JUMP_COOLDOWN_S` | 0.55 | ジャンプ再発火までの待機秒数 |

---

## 全身ミラー（`PoseAvatarDriver.cs`）

TimmyRobot は **Humanoid リグ**（`animationType: 3`、`optimizeGameObjects: 0` でボーンが Transform 露出）。
`Animator.GetBoneTransform(HumanBodyBones.X)` で各ボーンへアクセスし、`LateUpdate`（Animator 評価後）で
回転を上書きするため、ミラーが既製アニメに対して常に勝つ。

### 駆動ボーンと対応キーポイント

| ボーン | 子（方向計測用） | 始点kp → 終点kp |
|--------|-----------------|----------------|
| Spine | Chest/Neck | 腰中点 → 肩中点 |
| Neck | Head | 肩中点 → 鼻 |
| L/R UpperArm | LowerArm | 肩 → 肘 |
| L/R LowerArm | Hand | 肘 → 手首 |
| L/R UpperLeg | LowerLeg | 腰 → 膝 |
| L/R LowerLeg | Foot | 膝 → 足首 |

※ Hips（ルート回転）はミラー対象外（旋回ロコモーションとの競合回避）。

### リターゲット手順（各ボーン、親→子の順）

1. 2D デルタ `d = kp[終点] - kp[始点]`（画像 x=右, y=下、`mirror` 時は x を反転）
2. キャラ前額面へ写像: `targetDir = transform.right * d.x + transform.up * (-d.y)`
   （プレーンをキャラの向きに固定 → 体を旋回するとミラー全体も一緒に回る）
3. `desired = Quaternion.FromToRotation(現在の子方向, targetDir) * bone.rotation`（twist は維持）
4. フレームレート非依存の Slerp で平滑化（`responsiveness`）
5. キーポイント信頼度 `< kpMinConf` のボーンは更新を保留し**前回値を保持**（破綻防止）

### 制約

YOLO は 2D のため、奥行き・体のねじれは取得できず、**カメラ前額面内の近似ミラー**となる。
カメラに正対するほど忠実。

### 主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `mirror` | true | 鏡像（x 反転） |
| `swapSides` | false | 左右キーポイント対応の入替 |
| `kpMinConf` | 0.4 | ボーン更新の最低信頼度（Python 側と整合） |
| `responsiveness` | 15 | 大きいほど追従が速い／小さいほど滑らか |

---

## ロコモーション（`ThirdPersonController` ポーズ移動モード）

`PoseInputReceiver` が起動時に `PoseLocomotion=true` をセットし、毎フレーム
`PoseForward`/`PoseTurn`/`jump` を渡す。`PoseMove()` の挙動:

- `turn` でキャラ自身を yaw 回転（`transform.Rotate(0, turn*TurnSpeed*Δt, 0)`、`TurnSpeed`=120）
- `forward` でキャラの**自前方**へ前進（速度 = `forward*MoveSpeed`、既存の加減速イージング流用）
- **歩行アニメは出さない**（Animator の Speed/MotionSpeed を 0 に維持 → 脚は常時ミラー）
- 重力・接地・ジャンプ（`JumpAndGravity`）は既存実装を流用
- カメラはマウス無しのため `LockCameraPosition=true`

---

## レイテンシ分析

| 処理段階 | 所要時間 |
|---------|---------|
| カメラキャプチャ（60fps） | 16.7 ms |
| YOLO11n 推論（GPU） | 8〜12 ms |
| 特徴計算（足踏み/向き/ジャンプ） | <0.5 ms |
| UDP 送信（ループバック） | <0.3 ms |
| Unity UDP 受信 | <0.1 ms |
| Unity Update→LateUpdate 反映 | 次フレーム（0〜16.7 ms） |
| **合計（最良）** | **約 26〜32 ms** |
| **合計（最悪）** | **約 43〜48 ms** |

キーボード操作の一般的なレイテンシ（30〜50 ms）と同等で、リアルタイム操作として許容範囲。

---

## 作成・変更ファイル一覧

```
pose_estimation/
  main.py            変更（正規化kp＋特徴を build_packet で送信）
  gesture_mapper.py  変更（足踏みforward / 体の向きturn / jump に再構成）
  pose_sender.py     変更（v2 パケット：flat kp ＋特徴、build_packet 追加）

MothionCapture/Assets/SourceFiles/Scripts/
  PoseAvatarDriver.cs   新規（全身ボーンミラー）
  PoseInputReceiver.cs  変更（v2 受信・kp 公開・forward/turn 振分け・ポーズ移動モード起動）
  ThirdPersonController.cs 変更（PoseLocomotion モード / PoseMove 追加）

エディタ作業:
  PlayerRobot.prefab  Robot GameObject に PoseInputReceiver / PoseAvatarDriver を追加
```

---

## リスクと対策

| リスク | 対策 |
|--------|------|
| キーポイントのジッター | EMA 平滑化（特徴）＋ Slerp 平滑化（ボーン）＋デッドゾーン |
| 低信頼でボーンが破綻 | `kpMinConf` 未満は前回値を保持 |
| ジャンプ誤検知 | 両手首同時＋3フレーム持続＋クールダウン |
| 旋回ジェスチャと体幹ミラーの競合 | Hips をミラー対象外にし、移動は足踏み＋向きで分離 |
| 人物未検出でキャラが止まらない | 未検出時にニュートラルパケット（confidence:0）を送信 |
| ポート競合 | バインド失敗時は自動無効化＋エラーログ。受信は単一コンポーネントに集約 |
| Python プロセス落ち | 500ms タイムアウトで Unity がゼロ入力へ、ミラーは姿勢保持 |
| 複数人が映り込む | バウンディングボックス最大の人物を自動選択（`select_primary`） |
| 2D の奥行き欠落 | カメラ正対前提の平面ミラーとして許容（既知の限界） |

---

## 実装フェーズ

### Phase 1: 通信基盤 ✅ 完了
UDP 送受信（`pose_sender.py` / `PoseInputReceiver.cs`）、疎通確認。

### Phase 2: ジェスチャー認識（移動操作） ✅ 完了
`gesture_mapper.py` を足踏み forward / 体の向き turn / ジャンプに再構成、`[g]` デバッグ出力。

### Phase 3: 全身ミラー ✅ 完了
v2 パケット（flat kp）化、`PoseAvatarDriver.cs` による 2D→ボーン平面リターゲット。

### Phase 4: ロコモーション統合 ✅ 完了
`ThirdPersonController` ポーズ移動モード（足踏み前進＋向き旋回、歩行アニメ非再生）。

### Phase 5: 3D 化（MediaPipe） ✅ 完了
推論を MediaPipe Pose に置換、v3 パケットで `kp3d`（3D ワールド座標）を追加送信。
`PoseAvatarDriver.cs` を 3D ボーン方向（`useDepth`/`depthScale`、2D フォールバック付き）
に拡張し、奥行き方向の動きをミラー可能にした。

### Phase 5: 調整・運用（任意）
- `mirror`/`swapSides`/`responsiveness`/`TurnSpeed` と Python 側定数の実機調整
- デバッグ HUD（forward/turn/confidence 表示）
- 3D 姿勢推定への置換（奥行き対応、要 Python 大改修）
