using System.Collections.Generic;
using UnityEngine;

namespace PoseControl
{
    /// <summary>
    /// Full-body pose mirroring: retargets the MediaPipe keypoints received by
    /// <see cref="PoseInputReceiver"/> onto the Humanoid avatar's bones.
    ///
    /// Prefers the 3D world keypoints (kp3d, schema v3) so bones are oriented in
    /// full 3D — limbs can reach toward/away from the camera, not just within the
    /// frontal plane. Falls back to the 2D planar path when depth is unavailable.
    /// Rotations are applied in LateUpdate, after the Animator has evaluated the
    /// base pose, so the mirror always wins. Locomotion (root translation + turn
    /// + jump) is handled separately by ThirdPersonController; this component only
    /// orients limbs/torso/head.
    /// </summary>
    [DefaultExecutionOrder(200)]
    public class PoseAvatarDriver : MonoBehaviour
    {
        [Header("References")]
        [Tooltip("Source of keypoints. Auto-found on this GameObject if left empty.")]
        public PoseInputReceiver receiver;
        [Tooltip("Humanoid Animator to drive. Auto-found on this GameObject if left empty.")]
        public Animator animator;

        [Header("Mirroring")]
        [Tooltip("Reflect the pose horizontally (mirror-image feel).")]
        public bool mirror = true;
        [Tooltip("Swap left/right keypoint sources (use if limbs map to the wrong side).")]
        public bool swapSides = false;
        [Tooltip("Minimum keypoint confidence to update a bone; below this the bone holds its last pose.")]
        [Range(0f, 1f)] public float kpMinConf = 0.4f;
        [Tooltip("Higher = snappier limb tracking, lower = smoother/laggier.")]
        public float responsiveness = 15.0f;

        [Header("Depth (3D)")]
        [Tooltip("Use the 3D world keypoints (kp3d) for true depth. Off = planar 2D fallback.")]
        public bool useDepth = true;
        [Tooltip("Sign/scale for the depth axis. Flip the sign if reaching toward the camera bends limbs the wrong way.")]
        public float depthScale = -1.0f;

        // COCO keypoint indices
        const int NOSE = 0;
        const int L_SHO = 5, R_SHO = 6, L_ELB = 7, R_ELB = 8, L_WRI = 9, R_WRI = 10;
        const int L_HIP = 11, R_HIP = 12, L_KNE = 13, R_KNE = 14, L_ANK = 15, R_ANK = 16;
        const int N_KP = 17;

        struct RuntimeBone
        {
            public Transform bone;
            public Transform child;
            public int[] a;   // keypoint indices averaged for the head endpoint
            public int[] b;   // keypoint indices averaged for the tail endpoint
        }

        private readonly List<RuntimeBone> _bones = new List<RuntimeBone>();
        private Quaternion[] _smoothed;
        private bool[] _hasSmoothed;

        private void Start()
        {
            if (receiver == null) receiver = GetComponent<PoseInputReceiver>();
            if (animator == null) animator = GetComponent<Animator>();

            if (animator == null || !animator.isHuman)
            {
                Debug.LogError("[PoseAvatarDriver] Requires a Humanoid Animator. Disabling.");
                enabled = false;
                return;
            }

            // Torso & head, then each limb parent->child (order matters: parents first).
            Transform spineChild = Bone(HumanBodyBones.Chest)
                                   ?? Bone(HumanBodyBones.Neck)
                                   ?? Bone(HumanBodyBones.Head);
            Add(HumanBodyBones.Spine, spineChild, new[] { L_HIP, R_HIP }, new[] { L_SHO, R_SHO });
            Add(HumanBodyBones.Neck, Bone(HumanBodyBones.Head), new[] { L_SHO, R_SHO }, new[] { NOSE });

            Add(HumanBodyBones.LeftUpperArm, Bone(HumanBodyBones.LeftLowerArm), new[] { L_SHO }, new[] { L_ELB });
            Add(HumanBodyBones.LeftLowerArm, Bone(HumanBodyBones.LeftHand), new[] { L_ELB }, new[] { L_WRI });
            Add(HumanBodyBones.RightUpperArm, Bone(HumanBodyBones.RightLowerArm), new[] { R_SHO }, new[] { R_ELB });
            Add(HumanBodyBones.RightLowerArm, Bone(HumanBodyBones.RightHand), new[] { R_ELB }, new[] { R_WRI });

            Add(HumanBodyBones.LeftUpperLeg, Bone(HumanBodyBones.LeftLowerLeg), new[] { L_HIP }, new[] { L_KNE });
            Add(HumanBodyBones.LeftLowerLeg, Bone(HumanBodyBones.LeftFoot), new[] { L_KNE }, new[] { L_ANK });
            Add(HumanBodyBones.RightUpperLeg, Bone(HumanBodyBones.RightLowerLeg), new[] { R_HIP }, new[] { R_KNE });
            Add(HumanBodyBones.RightLowerLeg, Bone(HumanBodyBones.RightFoot), new[] { R_KNE }, new[] { R_ANK });

            _smoothed = new Quaternion[_bones.Count];
            _hasSmoothed = new bool[_bones.Count];
        }

        private Transform Bone(HumanBodyBones b) => animator.GetBoneTransform(b);

        private void Add(HumanBodyBones bone, Transform child, int[] a, int[] b)
        {
            Transform t = Bone(bone);
            if (t == null || child == null) return;  // rig lacks this bone; skip
            _bones.Add(new RuntimeBone { bone = t, child = child, a = a, b = b });
        }

        private void LateUpdate()
        {
            if (receiver == null) return;

            float[] kp = receiver.LatestKp;
            float[] kp3d = receiver.LatestKp3D;
            float[] conf = receiver.LatestKpConf;
            bool confOk = receiver.HasValidPose && conf != null && conf.Length >= N_KP;
            bool has3d = confOk && useDepth && kp3d != null && kp3d.Length >= N_KP * 3;
            bool has2d = confOk && kp != null && kp.Length >= N_KP * 2;

            float t = 1f - Mathf.Exp(-responsiveness * Time.deltaTime);  // framerate-independent slerp

            for (int i = 0; i < _bones.Count; i++)
            {
                RuntimeBone rb = _bones[i];

                bool gotTarget = false;
                Vector3 tgt = Vector3.zero;

                // Prefer the true 3D direction; fall back to the planar 2D one.
                if (has3d &&
                    TryPoint3D(kp3d, conf, rb.a, out Vector3 a3) &&
                    TryPoint3D(kp3d, conf, rb.b, out Vector3 b3))
                {
                    tgt = ToWorldDir3D(b3 - a3);
                    gotTarget = true;
                }
                else if (has2d &&
                    TryPoint(kp, conf, rb.a, out Vector2 pa) &&
                    TryPoint(kp, conf, rb.b, out Vector2 pb))
                {
                    tgt = ToWorldDir(pb - pa);
                    gotTarget = true;
                }

                if (gotTarget && IsFinite(tgt))
                {
                    Vector3 cur = rb.child.position - rb.bone.position;
                    if (cur.sqrMagnitude > 1e-8f && tgt.sqrMagnitude > 1e-8f)
                    {
                        Quaternion desired = Quaternion.FromToRotation(cur, tgt) * rb.bone.rotation;
                        if (IsFinite(desired))
                        {
                            _smoothed[i] = _hasSmoothed[i] ? Quaternion.Slerp(_smoothed[i], desired, t) : desired;
                            _hasSmoothed[i] = true;
                        }
                    }
                }

                // Apply the held rotation (last good pose) whenever we have one.
                // Guard against NaN/Inf so a single bad value can't blow up the whole rig.
                if (_hasSmoothed[i] && IsFinite(_smoothed[i]))
                    rb.bone.rotation = _smoothed[i];
            }
        }

        private static bool IsFinite(Vector3 v) =>
            !(float.IsNaN(v.x) || float.IsNaN(v.y) || float.IsNaN(v.z) ||
              float.IsInfinity(v.x) || float.IsInfinity(v.y) || float.IsInfinity(v.z));

        private static bool IsFinite(Quaternion q) =>
            !(float.IsNaN(q.x) || float.IsNaN(q.y) || float.IsNaN(q.z) || float.IsNaN(q.w) ||
              float.IsInfinity(q.x) || float.IsInfinity(q.y) || float.IsInfinity(q.z) || float.IsInfinity(q.w));

        /// <summary>Average the given keypoints into a 3D world point; false if any is low-confidence.</summary>
        private bool TryPoint3D(float[] kp3d, float[] conf, int[] idx, out Vector3 p)
        {
            Vector3 sum = Vector3.zero;
            for (int k = 0; k < idx.Length; k++)
            {
                int i = swapSides ? SwapSide(idx[k]) : idx[k];
                if (conf[i] < kpMinConf) { p = Vector3.zero; return false; }
                sum += new Vector3(kp3d[i * 3], kp3d[i * 3 + 1], kp3d[i * 3 + 2]);
            }
            p = sum / idx.Length;
            return true;
        }

        /// <summary>Map a 3D world delta (x right, y down, z depth) into the character's local axes.</summary>
        private Vector3 ToWorldDir3D(Vector3 d)
        {
            float x = mirror ? -d.x : d.x;
            return transform.right * x
                 + transform.up * (-d.y)
                 + transform.forward * (d.z * depthScale);
        }

        /// <summary>Average the given keypoints into a normalized image point; false if any is low-confidence.</summary>
        private bool TryPoint(float[] kp, float[] conf, int[] idx, out Vector2 p)
        {
            Vector2 sum = Vector2.zero;
            for (int k = 0; k < idx.Length; k++)
            {
                int i = swapSides ? SwapSide(idx[k]) : idx[k];
                if (conf[i] < kpMinConf) { p = Vector2.zero; return false; }
                float x = kp[i * 2];
                float y = kp[i * 2 + 1];
                if (mirror) x = 1f - x;
                sum += new Vector2(x, y);
            }
            p = sum / idx.Length;
            return true;
        }

        /// <summary>Map a 2D image delta (x right, y down) into the character's frontal plane.</summary>
        private Vector3 ToWorldDir(Vector2 d)
        {
            // Image y points down; avatar up is +Y. Keep the plane attached to the
            // character so turning the body rotates the whole mirrored pose with it.
            return transform.right * d.x + transform.up * (-d.y);
        }

        private static int SwapSide(int i)
        {
            switch (i)
            {
                case L_SHO: return R_SHO; case R_SHO: return L_SHO;
                case L_ELB: return R_ELB; case R_ELB: return L_ELB;
                case L_WRI: return R_WRI; case R_WRI: return L_WRI;
                case L_HIP: return R_HIP; case R_HIP: return L_HIP;
                case L_KNE: return R_KNE; case R_KNE: return L_KNE;
                case L_ANK: return R_ANK; case R_ANK: return L_ANK;
                case 1: return 2; case 2: return 1;  // eyes
                case 3: return 4; case 4: return 3;  // ears
                default: return i;
            }
        }
    }
}
