# yolo_onnx_ros public API changes (clang-tidy cleanup)

> **Who needs this:** anyone maintaining a package that includes
> `yolo_onnx_ros/detection.hpp` or `yolo_onnx_ros/yolo_inference.hpp`. At the
> moment that is `ed_sensor_integration` (`sam_seg_module.cpp`).
>
> These changes landed while bringing `yolo_onnx_ros` to zero clang-tidy
> findings during the ROS 2 migration. They are **source-breaking**: a
> dependent package will not compile until it is updated. None of them change
> runtime behaviour.

---

## 0. Find your call sites first

Run this from the root of your package. Everything it reports needs an edit.

```bash
grep -rn --include=*.h --include=*.hpp --include=*.cpp -E \
  '\b(YOLO_V8|DL_INIT_PARAM|DL_RESULT|_DL_INIT_PARAM|_DL_RESULT|MODEL_TYPE)\b|\b(YOLO_DETECT_V8|YOLO_POSE|YOLO_CLS)(_HALF|_V8_HALF)?\b|\bk(Onnx|TensorRT)\b|\b(Initialize|Detector|ReadYaml|CreateSession|RunSession|PreProcess)\s*\(|YOLO_ONNX_(TRUE|FALSE)\b' .
```

`sam_onnx_ros` uses the same names inside its `SEG::` namespace
(`SEG::DL_INIT_PARAM`, `SEG::DL_RESULT`, `SEG::Backend::kOnnx`, and the global
`Initialize(encoder, decoder, backend)` overload). The grep reports those too.
They were **not** renamed, so leave them alone.

### Coming from before `YoloWrapper`

`ed_sensor_integration`'s `master` predates `YoloWrapper` and does not compile
against `yolo_onnx_ros`'s `master` either. If your code still has
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

## 1. Renamed types

`readability-identifier-naming` requires `CamelCase` for classes. Pure renames.

| Was | Now | Declared in |
| --- | --- | --- |
| `YOLO_V8` | `YoloV8` | `yolo_onnx_ros/yolo_inference.hpp` |
| `DL_INIT_PARAM` | `DlInitParam` | `yolo_onnx_ros/yolo_inference.hpp` |
| `DL_RESULT` | `DlResult` | `yolo_onnx_ros/yolo_inference.hpp` |
| `MODEL_TYPE` | `ModelType` (now an `enum class`, see §3) | `yolo_onnx_ros/yolo_inference.hpp` |

`DlInitParam` and `DlResult` are now plain `struct`s instead of
`typedef struct _DL_INIT_PARAM {...} DL_INIT_PARAM;`. The `_DL_INIT_PARAM` and
`_DL_RESULT` names (reserved identifiers) no longer exist. Fields are unchanged,
and `DlResult::classId`/`confidence` are now zero-initialised.

## 2. Renamed functions

`camelBack` for functions. Same arguments, same semantics.

| Was | Now | Declared in |
| --- | --- | --- |
| `Initialize(model, backend)` | `initialize(model, backend)` | `yolo_onnx_ros/detection.hpp` |
| `Detector(wrapper, img)` | `detector(wrapper, img)` | `yolo_onnx_ros/detection.hpp` |
| `ReadYaml(file, p)` | `readYaml(file, p)` | `yolo_onnx_ros/detection.hpp` |
| `YOLO_V8::CreateSession()` | `YoloV8::createSession()` | `yolo_onnx_ros/yolo_inference.hpp` |
| `YOLO_V8::RunSession()` | `YoloV8::runSession()` | `yolo_onnx_ros/yolo_inference.hpp` |
| `YOLO_V8::PreProcess()` | `YoloV8::preProcess()` | `yolo_onnx_ros/yolo_inference.hpp` |

`preProcess()` now returns `const char*` instead of `char*`, like
`createSession()` and `runSession()` already did. Store the result in a
`const char*`.

## 3. Scoped enums

| Was | Now |
| --- | --- |
| `YOLO::Backend::kOnnx` | `YOLO::Backend::ONNX` |
| `YOLO::Backend::kTensorRT` | `YOLO::Backend::TENSORRT` |
| `YOLO_DETECT_V8`, `YOLO_POSE`, `YOLO_CLS`, `YOLO_DETECT_V8_HALF`, `YOLO_POSE_V8_HALF`, `YOLO_CLS_HALF` | `ModelType::YOLO_DETECT_V8`, … (same names, now scoped) |

`ModelType` is an `enum class`, so the values must be qualified and no longer
convert to `int` implicitly. Code like `if (params.modelType < 4)` (FP32 vs
FP16 models) needs an explicit comparison against the enumerators. Both enums
now have `std::uint8_t` as their underlying type.

## 4. `config.hpp`

`YOLO_ONNX_ROS_CUDA_ENABLED` and `YOLO_ONNX_ROS_TENSORRT_ENABLED` are still
`0`/`1`, so `#if YOLO_ONNX_ROS_TENSORRT_ENABLED` keeps working. The helper
macros `YOLO_ONNX_TRUE`/`YOLO_ONNX_FALSE` are gone.

---

## 5. Mechanical migration

For most code, this `sed` handles §1–§3. It only rewrites unqualified names and
`YOLO::` names, so `SEG::` names from `sam_onnx_ros` are left untouched. It
can't tell the two apart behind a `using namespace SEG;`, so review the diff.

```bash
grep -rlE --include=*.h --include=*.hpp --include=*.cpp \
  'YOLO_V8|DL_INIT_PARAM|DL_RESULT|MODEL_TYPE|kOnnx|kTensorRT|YOLO_(DETECT|POSE|CLS)|Detector\(|ReadYaml|CreateSession|RunSession|PreProcess' . |
xargs sed -i -E \
  -e 's/(^|[^:A-Za-z0-9_])YOLO_V8\b/\1YoloV8/g' \
  -e 's/(^|[^:A-Za-z0-9_])DL_INIT_PARAM\b/\1DlInitParam/g' \
  -e 's/(^|[^:A-Za-z0-9_])DL_RESULT\b/\1DlResult/g' \
  -e 's/(^|[^:A-Za-z0-9_])MODEL_TYPE\b/\1ModelType/g' \
  -e 's/YOLO::Backend::kOnnx\b/YOLO::Backend::ONNX/g' \
  -e 's/YOLO::Backend::kTensorRT\b/YOLO::Backend::TENSORRT/g' \
  -e 's/(^|[^:A-Za-z0-9_])(YOLO_(DETECT_V8|POSE|CLS)(_HALF|_V8_HALF)?)\b/\1ModelType::\2/g' \
  -e 's/(^|[^:A-Za-z0-9_])Detector\(/\1detector(/g' \
  -e 's/(^|[^:A-Za-z0-9_])ReadYaml\(/\1readYaml(/g' \
  -e 's/(->|\.)CreateSession\(/\1createSession(/g' \
  -e 's/(->|\.)RunSession\(/\1runSession(/g' \
  -e 's/(->|\.)PreProcess\(/\1preProcess(/g'
```

`Initialize(` is deliberately left out, because `sam_onnx_ros` uses the same
name. Rename the YOLO calls to `initialize(` by hand.

---

## 6. Packaging changes (not source-breaking)

- `yolo_onnx_ros_node` is installed to `lib/yolo_onnx_ros`, so
  `ros2 run yolo_onnx_ros yolo_onnx_ros_node` finds it.
- With TensorRT enabled, `tensorrt_ros` and the vendored `yolos` headers are now
  part of the exported link interface, so consumers that include
  `detection.hpp` compile without extra include directories.
