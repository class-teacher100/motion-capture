# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Structure

This is a monorepo containing two projects that work together as a real-time
motion-capture pipeline:

1. **pose_estimation/** - Real-time pose estimation application (Python)
2. **MothionCapture/** - Unity game project with character controller

The Python app streams pose data over UDP to the Unity app, which mirrors the
user's whole body onto a Humanoid robot and drives intentional locomotion. See
**Pose Control Integration** below and `doc/implementation-plan.md` for details.

## Project 1: pose_estimation (Python)

### Tech Stack
- **Language**: Python 3.12
- **Package Manager**: uv (fast Python package manager)
- **ML Framework**: MediaPipe Pose (3D landmarks, runs on CPU)
- **Vision**: OpenCV
- **Dependencies**: mediapipe, opencv-python, numpy

### Setup & Running

Install dependencies (includes auto-setup of Python 3.12):
```powershell
uv sync
```

Run the application:
```powershell
uv run python main.py
```

The first run auto-downloads the MediaPipe Pose model.

### Customization

Edit constants at the top of `main.py`:
- `KP_CONF_THRESHOLD` (0.5) - Keypoint confidence (MediaPipe visibility) threshold
- `MODEL_VARIANT` ("full") - "lite" (fastest) / "full" (balanced) / "heavy" (most accurate)
- `MIN_DET_CONF` / `MIN_PRESENCE_CONF` / `MIN_TRACK_CONF` (0.5) - Detection / presence / tracking confidence
- `CAMERA_INDEX` (0) - Camera selection for multi-camera setups

Tune locomotion gestures in `gesture_mapper.py` (e.g. `STEP_MIN_AMP`, `STEP_FULL_CROSS`
for forward sensitivity; `TURN_DEADZONE`, `TURN_SCALE` for turning; `JUMP_HOLD_FRAMES`).

### Architecture

**main.py** is the entry point. Core flow:
1. Creates a MediaPipe Tasks PoseLandmarker (CPU; first run downloads the `.task` model)
2. Opens camera and captures frames
3. For each frame: runs inference → remaps MediaPipe's 33 landmarks to the 17
   COCO keypoints (2D normalized + visibility + 3D world landmarks) → draws skeleton
4. Derives locomotion features and streams a v3 UDP packet (2D + 3D) to Unity
5. Displays FPS, detection state, and model info on frame
6. Runtime controls: `q` quit, `p` print keypoints, `g` print locomotion features

**Modules**:
- `gesture_mapper.py` - `GestureMapper` converts 2D keypoints (pixel coords) into
  locomotion features: `forward` (step-in-place cadence), `turn` (upper-body
  orientation), `jump` (both wrists raised). Unchanged by the 3D upgrade.
- `pose_sender.py` - `PoseSender` (UDP localhost:5005) + `build_packet()` (v3 schema:
  flat normalized `kp[34]` + `kp_conf[17]` + `kp3d[51]` + features)

**Key Functions** (main.py):
- `extract_coco()` - Remaps MediaPipe's 33 landmarks to the 17 COCO keypoints,
  returning 2D normalized coords, visibility, and 3D world landmarks
- `draw_pose()` - Renders green keypoint circles and orange skeleton lines
- `print_pose_data()` - Outputs 2D + 3D keypoint positions when `p` is pressed

MediaPipe tracks a single person; the largest/most-confident body in frame is used
(no multi-person selection needed).

### Detected Keypoints

The pipeline uses the 17 COCO-standard keypoints (nose, eyes, ears, shoulders,
elbows, wrists, hips, knees, ankles) — a subset of MediaPipe's 33 landmarks, each
now carrying a 3D (x, y, z) world position in addition to 2D. See `COCO_FROM_MP`
mapping and skeleton connectivity in main.py.

---

## Project 2: MothionCapture (Unity)

### Tech Stack
- **Engine**: Unity 6000.4.7f1 (LTS)
- **Language**: C# (Unity Scripts)
- **Rendering**: Universal Render Pipeline (URP) 17.4.0
- **Input**: New Input System 1.19.0
- **Cameras**: Cinemachine 3.1.6
- **Audio**: Native Unity AudioSource

### Project Location
`MothionCapture/` - Standard Unity project structure

### Assets Organization

**Custom Scripts** (`Assets/SourceFiles/`):
- `Scripts/` - Core gameplay logic
- `InputSystem/` - Input handling via New Input System
- `Animation/`, `Models/`, `Materials/`, `Textures/` - Game assets
- `SoundFX/` - Audio files
- `Prefabs/`, `Scenes/` - Game scenes and prefabs

### Key Scripts

**ThirdPersonController.cs** - Main character controller
- Handles movement (WASD), sprint, jumping
- Implements gravity and ground detection
- Manages footstep/landing audio via `FootstepAudioClips[]`
- Camera control with Cinemachine integration
- Customizable movement speed (2.0 m/s), sprint speed (5.335 m/s)
- **Pose locomotion mode** (`PoseLocomotion`): when enabled, `PoseMove()` drives the
  root from `PoseForward`/`PoseTurn` (step-forward + body-turn) instead of camera-relative
  WASD, suppresses the walk animation (body is mirrored instead), and reuses gravity/jump

**StarterAssetsInputs.cs** - Input adapter
- Maps New Input System callbacks to character state (move, look, jump, sprint)
- Handles cursor lock/visibility

**PoseInputReceiver.cs** (`PoseControl`) - UDP pose receiver (single bind, port 5005)
- Background thread receives v2 packets; `Update()` feeds locomotion to ThirdPersonController
- Exposes latest `LatestKp`/`LatestKpConf`/`HasValidPose` for the mirroring layer
- On start, enables pose mode (disables PlayerInput, locks camera, sets `PoseLocomotion`)

**PoseAvatarDriver.cs** (`PoseControl`) - Full-body pose mirroring
- In `LateUpdate` (after Animator), retargets the 3D world keypoints (`kp3d`) onto
  Humanoid bones (spine, neck, both arms and legs) via `Animator.GetBoneTransform` +
  `FromToRotation`; falls back to the 2D planar path when depth is unavailable
- 3D avateering with confidence gating, Slerp smoothing, `mirror`/`swapSides`, and
  `useDepth`/`depthScale` (flip `depthScale` sign if reach-toward-camera bends the wrong way)

**MotionAudioController.cs** - Movement-triggered audio
- Monitors character position changes
- Plays ambient audio when moving, fades out when stationary
- Uses coroutine-based volume lerp for smooth fade-out

### Dependencies

Key packages (from `Packages/manifest.json`):
- **com.unity.cinemachine** - Cinematic camera system
- **com.unity.inputsystem** - New input handling
- **com.unity.render-pipelines.universal** - URP graphics
- **com.unity.learn.iet-framework** - Tutorial framework (for onboarding)

### Architecture Notes

- **Third-person perspective** - Player character with offset camera controlled by Cinemachine
- **Character physics** - CharacterController component (not Rigidbody)
- **Audio system** - One-shot footsteps on animation events + continuous motion-triggered ambience
- **No multiplayer logic** - Single-player focused despite multiplayer.center package

### Running/Building

Open the project in Unity Editor 6000.4.7f1. Main scene is loaded via `LastSceneManagerSetup.txt`. Edit scenes in the editor or build as Windows executable via File > Build Settings.

---

## Pose Control Integration (Python ↔ Unity)

Two-layer design sharing one UDP datagram (`127.0.0.1:5005`, JSON):

- **Mirror layer** — full-body avateering. The 3D world keypoints are retargeted onto the
  Humanoid robot's bones (`PoseAvatarDriver`), so limbs can reach toward/away from the
  camera, not just within the frontal plane. Falls back to the 2D planar path if depth
  is unavailable. Most faithful when the user faces the camera.
- **Locomotion layer** — intentional movement. `forward`/`turn`/`jump` drive root motion
  via `ThirdPersonController`'s pose mode (step-in-place = forward, body twist = turn).

**Packet schema v3** (`pose_sender.build_packet`):
```json
{"v":3, "kp":[x0,y0,...x16,y16], "kp_conf":[...17], "kp3d":[x0,y0,z0,...x16,y16,z16], "forward":0..1, "turn":-1..1, "jump":bool, "confidence":0..1}
```
All arrays are flat floats (Unity `JsonUtility` cannot parse jagged arrays). `kp` is
2D normalized image coords; `kp3d` is metric, hip-centered 3D world landmarks (51 floats).
When `confidence < 0.4` or packets stop for 0.5 s, locomotion zeroes out and the mirror
holds its last good pose.

**Avatar**: `Assets/Prefabs/PlayerRobot.prefab` → `Robot` GameObject (Humanoid rig,
`TimmyRobot.fbx`, `animationType: 3`). The two `PoseControl` components are attached here
alongside `Animator`/`ThirdPersonController`/`StarterAssetsInputs`. Both auto-resolve their
references from the same GameObject. See `doc/usage-guide.md` for editor setup steps.

Keep the Python feature constants and Unity inspector fields in sync (e.g. `KP_MIN_CONF` ≈
`PoseAvatarDriver.kpMinConf`; `udpPort` ≈ `pose_sender.UDP_PORT`).

---

## Development Workflow

### Combined (pose control end-to-end)
1. Start Python first: `uv run python main.py` (stand so head-to-ankles are in frame)
2. Press Play in Unity; console should log `[PoseInputReceiver] Listening on UDP port 5005`
3. Tune: `mirror`/`swapSides`/`responsiveness` (PoseAvatarDriver), `TurnSpeed`
   (ThirdPersonController), and `STEP_*`/`TURN_*` (gesture_mapper.py)

### Python (pose_estimation)
1. Activate environment: `uv sync`
2. Run: `uv run python main.py`
3. Test changes by modifying constants in main.py / gesture_mapper.py and re-running
4. Runs on CPU (MediaPipe); no GPU/CUDA required. Raise `MODEL_COMPLEXITY` for accuracy.

### Unity (MothionCapture)
1. Open project in Unity 6000.4.7f1
2. Edit scripts in `Assets/SourceFiles/Scripts/`
3. Play in editor to test (Ctrl+P or Play button)
4. Changes to C# scripts auto-recompile and hot-reload

### Key Directories to Know
- `pose_estimation/` - Python pose app (camera → features/keypoints → UDP)
- `MothionCapture/Assets/SourceFiles/Scripts/` - Custom game + pose-control code
- `MothionCapture/Packages/` - Unity package dependencies
- `MothionCapture/ProjectSettings/` - Unity project configuration
- `doc/` - Design docs (`implementation-plan.md`, `usage-guide.md`)

