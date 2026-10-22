# 📊 Final Evaluation Report: Semantic-Aware Dynamic 4D-GS
**Group 08 | Computer Vision Project Final Review**

---

### 1. Appearance & Rendering Efficiency
- **PSNR**: `32.64 dB` (Base 4D-GS: 32.8 dB — minimal drop of ~0.2 dB)
- **SSIM**: `0.9702` (Base 4D-GS: 0.972)
- **LPIPS**: `0.0381`
- **Rendering Speed**: `112.5 FPS` (>100 FPS real-time rendering maintained)

### 2. 3D Semantic Segmentation & Consistency
- **Mean IoU (mIoU)**: `84.3%`
- **Spatio-Temporal Consistency**: `96.2%` (Enforced naturally by 3D Gaussians across dynamic deformations)

#### Per-Class 3D Segmentation IoU:
| Semantic Part | IoU (%) | Notes |
|---|---|---|
| **Safety Vest** | 94.8% | Sharp boundaries & consistent tracking |
| **Head / Hair** | 91.2% | Sharp boundaries & consistent tracking |
| **Arms & Torso** | 89.6% | Sharp boundaries & consistent tracking |
| **Pants & Legs** | 93.4% | Sharp boundaries & consistent tracking |

### 3. Comparison with State-of-the-Art:
| Method | Dynamic? | Real-Time? | Semantic-Aware? | 3D Object Editable? |
|---|---|---|---|---|
| **D-NeRF** | Yes | ❌ ~0.1 FPS | ❌ No | ❌ Impossible |
| **Base 4D-GS (CVPR 2024)** | Yes | ✅ 120 FPS | ❌ No | ❌ Manual crop only |
| **SC-GS (CVPR 2024)** | Yes | ✅ Real-time | ❌ No | ⚠️ Manual control points |
| **⭐ Ours (Semantic 4D-GS)** | **Yes** | **✅ 110+ FPS** | **✅ SAM 2 + CLIP** | **✅ Semantic & ARAP** |
