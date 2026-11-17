# yolo_onnx_ros and sam_onnx_ros public API changes (ROS 2 migration)

> **Who needs this:** anyone maintaining a package that includes
> `yolo_onnx_ros/*.hpp` or `sam_onnx_ros/*.hpp`. At the moment that is
> `ed_sensor_integration` (`sam_seg_module.cpp`), which uses both.
>
> Both packages were brought to zero clang-tidy findings during their ROS 2
> migration (`yolo_onnx_ros` #15, `sam_onnx_ros` #11), and both renamed their
> public API to the tue-robotics naming style the same way. The changes are
> **source-breaking**: a dependent package will not compile until it is
> updated. The renames themselves don't change runtime behaviour.
>
> The ROS 1 versions stay available on each repo's `ros1` branch.

---

## 0. Find your call sites first

Run this from the root of your package. Everything it reports needs an edit.

```bash
grep -rn --include=*.h --include=*.hpp --include=*.cpp -E \
  '\b(YOLO_V8|DL_INIT_PARAM|DL_RESULT|_DL_INIT_PARAM|_DL_RESULT|MODEL_TYPE)\b|\b(YOLO_DETECT_V8|YOLO_POSE|YOLO_CLS)(_HALF|_V8_HALF)?\b|\bSAM_SEGMENT_(ENCODER|DECODER)\b|\bk(Onnx|TensorRT|SpeedSam)\b|\b(Initialize|Detector|ReadYaml|SegmentAnything|CreateSession|RunSession|PreProcess|ScaleBboxPoints|PrepareInputTensor|PostProcess|BlobFromImage)\s*\(|(YOLO|SAM)_ONNX_(TRUE|FALSE)\b' .
```

### Coming from before `YoloWrapper`

`ed_sensor_integration`'s `master` predates `YoloWrapper` and does not compile
against `yolo_onnx_ros`'s ROS 1 `master` either. If your code still has
`std::unique_ptr<YOLO_V8>` as the detector returned by `Initialize()`, first
switch to `YoloWrapper` (as `ed_sensor_integration`'s `subfeature/sam_tensor_rt`
branch already does), then apply the renames below:

```c++
// Before YoloWrapper
std::unique_ptr<YOLO_V8> yoloDetector;
std::tie(yoloDetector, yolo_params) = Initialize(yolo_model);
resYolo = Detector(yoloDetector, img);

// Now
YoloWrapper yolo_wrapper;
std::tie(yolo_wrapper, yolo_params) = initialize(yolo_model, YOLO::Backend::ONNX);
resYolo = detector(yolo_wrapper, img);
// yoloDetector->classes is now yolo_wrapper.classes
```

---

## 1. yolo_onnx_ros

### Types

`readability-identifier-naming` requires `CamelCase` for classes. Pure renames.

| Was | Now | Declared in |
| --- | --- | --- |
| `YOLO_V8` | `YoloV8` | `yolo_onnx_ros/yolo_inference.hpp` |
| `DL_INIT_PARAM` | `DlInitParam` | `yolo_onnx_ros/yolo_inference.hpp` |
| `DL_RESULT` | `DlResult` | `yolo_onnx_ros/yolo_inference.hpp` |
| `MODEL_TYPE` | `ModelType` (now an `enum class`) | `yolo_onnx_ros/yolo_inference.hpp` |

`DlInitParam` and `DlResult` are now plain `struct`s instead of
`typedef struct _DL_INIT_PARAM {...} DL_INIT_PARAM;`. The `_DL_INIT_PARAM` and
`_DL_RESULT` names (reserved identifiers) no longer exist. Fields are unchanged,
and `DlResult::classId`/`confidence` are now zero-initialised.

### Functions

`camelBack` for functions. Same arguments, same semantics.

| Was | Now | Declared in |
| --- | --- | --- |
| `Initialize(model, backend)` | `initialize(model, backend)` | `yolo_onnx_ros/detection.hpp` |
| `Detector(wrapper, img)` | `detector(wrapper, img)` | `yolo_onnx_ros/detection.hpp` |
| `ReadYaml(file, p)` | `readYaml(file, p)` | `yolo_onnx_ros/detection.hpp` |
| `YOLO_V8::CreateSession()` | `YoloV8::createSession()` | `yolo_onnx_ros/yolo_inference.hpp` |
| `YOLO_V8::RunSession()` | `YoloV8::runSession()` | `yolo_onnx_ros/yolo_inference.hpp` |
| `YOLO_V8::PreProcess()` | `YoloV8::preProcess()` | `yolo_onnx_ros/yolo_inference.hpp` |

`YoloV8::preProcess()` now returns `const char*` instead of `char*`, like
`createSession()` and `runSession()` already did. Store the result in a
`const char*`.

### Enums

| Was | Now |
| --- | --- |
| `YOLO::Backend::kOnnx` | `YOLO::Backend::ONNX` |
| `YOLO::Backend::kTensorRT` | `YOLO::Backend::TENSORRT` |
| `YOLO_DETECT_V8`, `YOLO_POSE`, `YOLO_CLS`, `YOLO_DETECT_V8_HALF`, `YOLO_POSE_V8_HALF`, `YOLO_CLS_HALF` | `ModelType::YOLO_DETECT_V8`, … (same names, now scoped) |

`ModelType` is an `enum class`, so the values must be qualified and no longer
convert to `int` implicitly. Code like `if (params.modelType < 4)` (FP32 vs
FP16 models) needs an explicit comparison against the enumerators. Both enums
now have `std::uint8_t` as their underlying type.

### `config.hpp`

`YOLO_ONNX_ROS_CUDA_ENABLED` and `YOLO_ONNX_ROS_TENSORRT_ENABLED` are still
`0`/`1`, so `#if YOLO_ONNX_ROS_TENSORRT_ENABLED` keeps working. The helper
macros `YOLO_ONNX_TRUE`/`YOLO_ONNX_FALSE` are gone.

---

## 2. sam_onnx_ros

Everything stays in the `SEG` namespace. Only the names inside it changed, in
the same way as in `yolo_onnx_ros`.

### Types

| Was | Now | Declared in |
| --- | --- | --- |
| `SEG::DL_INIT_PARAM` | `SEG::DlInitParam` | `sam_onnx_ros/dl_types.hpp` |
| `SEG::DL_RESULT` | `SEG::DlResult` | `sam_onnx_ros/dl_types.hpp` |
| `SEG::MODEL_TYPE` | `SEG::ModelType` (now an `enum class`) | `sam_onnx_ros/dl_types.hpp` |

`SAM`, `Utils` and `SamWrapper` keep their names. As in `yolo_onnx_ros`, the
`_DL_INIT_PARAM`/`_DL_RESULT` struct tags no longer exist. `operator<<` on
`SEG::DlInitParam` now prints `modelType` as its integer value.

### Functions

| Was | Now | Declared in |
| --- | --- | --- |
| `Initialize(encoder, decoder, backend)` | `initialize(encoder, decoder, backend)` | `sam_onnx_ros/segmentation.hpp` |
| `SegmentAnything(wrapper, enc, dec, img, resSam, res)` | `segmentAnything(wrapper, enc, dec, img, resSam, res)` | `sam_onnx_ros/segmentation.hpp` |
| `SAM::CreateSession()` | `SAM::createSession()` | `sam_onnx_ros/sam_inference.hpp` |
| `SAM::RunSession()` | `SAM::runSession()` | `sam_onnx_ros/sam_inference.hpp` |
| `Utils::PreProcess()` | `Utils::preProcess()` | `sam_onnx_ros/utils.hpp` |
| `Utils::ScaleBboxPoints()` | `Utils::scaleBboxPoints()` | `sam_onnx_ros/utils.hpp` |
| `Utils::PrepareInputTensor()` | `Utils::prepareInputTensor()` (now `static`) | `sam_onnx_ros/utils.hpp` |
| `Utils::PostProcess()` | `Utils::postProcess()` (now `static`) | `sam_onnx_ros/utils.hpp` |
| `Utils::BlobFromImage()` | `Utils::blobFromImage()` | `sam_onnx_ros/utils.hpp` |

`Utils::~Utils()` is now `= default`.

### Enums

| Was | Now |
| --- | --- |
| `SEG::Backend::kOnnx` | `SEG::Backend::ONNX` |
| `SEG::Backend::kSpeedSam` | `SEG::Backend::SPEED_SAM` |
| `SEG::SAM_SEGMENT_ENCODER`, `SEG::SAM_SEGMENT_DECODER` | `SEG::ModelType::SAM_SEGMENT_ENCODER`, `SEG::ModelType::SAM_SEGMENT_DECODER` |

---

## 3. Mechanical migration

For most code, this `sed` handles §1 and §2. Review the diff afterwards. It
renames `Initialize(`, `Detector(` and `SegmentAnything(` wherever they are
called as free functions, so check that your package has no unrelated function
with one of those names. An unqualified `SAM_SEGMENT_ENCODER` (inside
`namespace SEG` or behind `using namespace SEG;`) is not rewritten; qualify it
by hand.

```bash
grep -rlE --include=*.h --include=*.hpp --include=*.cpp \
  'YOLO_V8|DL_INIT_PARAM|DL_RESULT|MODEL_TYPE|SAM_SEGMENT_|kOnnx|kTensorRT|kSpeedSam|YOLO_(DETECT|POSE|CLS)|Initialize\(|Detector\(|ReadYaml|SegmentAnything|CreateSession|RunSession|PreProcess|ScaleBboxPoints|PrepareInputTensor|PostProcess|BlobFromImage' . |
xargs sed -i -E \
  -e 's/\bYOLO_V8\b/YoloV8/g' \
  -e 's/\bDL_INIT_PARAM\b/DlInitParam/g' \
  -e 's/\bDL_RESULT\b/DlResult/g' \
  -e 's/\bMODEL_TYPE\b/ModelType/g' \
  -e 's/Backend::kOnnx\b/Backend::ONNX/g' \
  -e 's/Backend::kTensorRT\b/Backend::TENSORRT/g' \
  -e 's/Backend::kSpeedSam\b/Backend::SPEED_SAM/g' \
  -e 's/(^|[^:A-Za-z0-9_])(YOLO_(DETECT_V8|POSE|CLS)(_HALF|_V8_HALF)?)\b/\1ModelType::\2/g' \
  -e 's/SEG::(SAM_SEGMENT_(ENCODER|DECODER))\b/SEG::ModelType::\1/g' \
  -e 's/(^|[^:.>A-Za-z0-9_])Initialize\(/\1initialize(/g' \
  -e 's/(^|[^:.>A-Za-z0-9_])Detector\(/\1detector(/g' \
  -e 's/(^|[^:.>A-Za-z0-9_])ReadYaml\(/\1readYaml(/g' \
  -e 's/(^|[^:.>A-Za-z0-9_])SegmentAnything\(/\1segmentAnything(/g' \
  -e 's/(->|\.|::)CreateSession\(/\1createSession(/g' \
  -e 's/(->|\.|::)RunSession\(/\1runSession(/g' \
  -e 's/(->|\.|::)PreProcess\(/\1preProcess(/g' \
  -e 's/(->|\.|::)ScaleBboxPoints\(/\1scaleBboxPoints(/g' \
  -e 's/(->|\.|::)PrepareInputTensor\(/\1prepareInputTensor(/g' \
  -e 's/(->|\.|::)PostProcess\(/\1postProcess(/g' \
  -e 's/(->|\.|::)BlobFromImage\(/\1blobFromImage(/g'
```

---

## 4. Building against them in ROS 2

Both packages are `ament_cmake` packages that export a namespaced CMake target,
replacing catkin's `LIBRARIES`:

```cmake
find_package(sam_onnx_ros REQUIRED)
find_package(yolo_onnx_ros REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE  # PUBLIC if your public headers include theirs
  sam_onnx_ros::sam_onnx_ros
  yolo_onnx_ros::yolo_onnx_ros
)
```

They bring their own dependencies (`onnxruntime_ros`, OpenCV, and CUDA or
TensorRT when enabled) through `ament_export_dependencies`, so you don't need
to list those yourself.

Other `yolo_onnx_ros` packaging changes:

- `yolo_onnx_ros_node` is installed to `lib/yolo_onnx_ros`, so
  `ros2 run yolo_onnx_ros yolo_onnx_ros_node` finds it.
- With TensorRT enabled, `tensorrt_ros` and the vendored `yolos` headers are now
  part of the exported link interface, so consumers that include
  `detection.hpp` compile without extra include directories.
