using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif
using StarterAssets;

namespace PoseControl
{
    // Packet schema v2 (see pose_estimation/pose_sender.py).
    // kp is a flat float[34] = [x0,y0, x1,y1, ... x16,y16] normalized 0..1.
    // JsonUtility supports float[] but not jagged arrays, hence the flat layout.
    [Serializable]
    struct PosePacket
    {
        public int v;
        public float[] kp;
        public float[] kp_conf;
        public float forward;
        public float turn;
        public bool jump;
        public float confidence;
    }

    // Run after PlayerInput (order 0) so pose data always wins over keyboard residuals.
    // Also runs before PoseAvatarDriver's LateUpdate so the shared keypoints are fresh.
    [DefaultExecutionOrder(100)]
    public class PoseInputReceiver : MonoBehaviour
    {
        [Header("Pose Control")]
        public bool poseControlEnabled = true;
        public int udpPort = 5005;
        public float minConfidence = 0.4f;
        public float packetTimeoutSeconds = 0.5f;

        [Header("References")]
        public StarterAssetsInputs starterAssetsInputs;

        // ── Shared with PoseAvatarDriver (read in its LateUpdate) ──────────────
        /// <summary>Flat normalized keypoints [x0,y0,...x16,y16], or null until first packet.</summary>
        public float[] LatestKp { get; private set; }
        /// <summary>Per-keypoint confidences (17), or null until first packet.</summary>
        public float[] LatestKpConf { get; private set; }
        /// <summary>True when the latest pose is fresh and confident enough to drive bones.</summary>
        public bool HasValidPose { get; private set; }

        private UdpClient _udpClient;
        private Thread _networkThread;
        private volatile bool _running;

        private readonly object _lock = new object();
        private PosePacket _latestPacket;
        private long _lastPacketMs = -1L;  // Stopwatch.ElapsedMilliseconds, -1 = never received
        private static readonly System.Diagnostics.Stopwatch _sw = System.Diagnostics.Stopwatch.StartNew();

#if ENABLE_INPUT_SYSTEM
        private PlayerInput _playerInput;
#endif
        private ThirdPersonController _thirdPersonController;

        private void Start()
        {
            if (starterAssetsInputs == null)
                starterAssetsInputs = GetComponent<StarterAssetsInputs>();

#if ENABLE_INPUT_SYSTEM
            _playerInput = GetComponent<PlayerInput>();
#endif
            _thirdPersonController = GetComponent<ThirdPersonController>();

            if (poseControlEnabled)
            {
                // Analog values must drive speed proportionally.
                if (starterAssetsInputs != null)
                    starterAssetsInputs.analogMovement = true;

                // Prevent keyboard/gamepad from overwriting pose input every frame.
#if ENABLE_INPUT_SYSTEM
                if (_playerInput != null)
                    _playerInput.enabled = false;
#endif

                if (_thirdPersonController != null)
                {
                    // Lock camera: pose control has no mouse, so camera must stay fixed.
                    _thirdPersonController.LockCameraPosition = true;
                    // Drive root with step-forward + body-turn instead of camera-relative WASD.
                    _thirdPersonController.PoseLocomotion = true;
                }
            }

            try
            {
                _udpClient = new UdpClient(udpPort);
                _running = true;
                _networkThread = new Thread(NetworkLoop)
                {
                    IsBackground = true,
                    Name = "PoseUDPReceiver"
                };
                _networkThread.Start();
                Debug.Log($"[PoseInputReceiver] Listening on UDP port {udpPort}");
            }
            catch (SocketException e)
            {
                Debug.LogError($"[PoseInputReceiver] Cannot bind port {udpPort}: {e.Message}. Pose control disabled.");
                poseControlEnabled = false;

                // Restore normal input on failure.
#if ENABLE_INPUT_SYSTEM
                if (_playerInput != null)
                    _playerInput.enabled = true;
#endif
                if (_thirdPersonController != null)
                {
                    _thirdPersonController.LockCameraPosition = false;
                    _thirdPersonController.PoseLocomotion = false;
                }
                if (starterAssetsInputs != null)
                    starterAssetsInputs.analogMovement = false;
            }
        }

        private void NetworkLoop()
        {
            var remoteEP = new IPEndPoint(IPAddress.Any, 0);
            while (_running)
            {
                try
                {
                    byte[] bytes = _udpClient.Receive(ref remoteEP);
                    string json = Encoding.UTF8.GetString(bytes);
                    PosePacket packet = JsonUtility.FromJson<PosePacket>(json);
                    lock (_lock)
                    {
                        _latestPacket = packet;
                        _lastPacketMs = _sw.ElapsedMilliseconds;
                    }
                }
                catch (SocketException)
                {
                    break;  // socket closed during shutdown
                }
                catch (Exception e)
                {
                    Debug.LogWarning($"[PoseInputReceiver] Packet parse error: {e.Message}");
                }
            }
        }

        private void Update()
        {
            if (!poseControlEnabled || starterAssetsInputs == null) return;

            PosePacket p;
            long lastMs;
            lock (_lock)
            {
                p = _latestPacket;
                lastMs = _lastPacketMs;
            }

            long elapsedMs = lastMs < 0 ? long.MaxValue : _sw.ElapsedMilliseconds - lastMs;
            bool timedOut = elapsedMs > (long)(packetTimeoutSeconds * 1000);
            bool valid = !timedOut && p.confidence >= minConfidence;

            // Expose keypoints for the mirroring layer (held when invalid so limbs freeze).
            if (valid)
            {
                LatestKp = p.kp;
                LatestKpConf = p.kp_conf;
            }
            HasValidPose = valid;

            // Locomotion layer.
            if (!valid)
            {
                starterAssetsInputs.JumpInput(false);
                if (_thirdPersonController != null)
                {
                    _thirdPersonController.PoseForward = 0f;
                    _thirdPersonController.PoseTurn = 0f;
                }
                return;
            }

            if (_thirdPersonController != null)
            {
                _thirdPersonController.PoseForward = Mathf.Clamp01(p.forward);
                _thirdPersonController.PoseTurn = Mathf.Clamp(p.turn, -1f, 1f);
            }
            starterAssetsInputs.JumpInput(p.jump);  // false also clears the latched flag
        }

        private void OnDestroy()
        {
            _running = false;
            _udpClient?.Close();
            _networkThread?.Join(500);
        }
    }
}
