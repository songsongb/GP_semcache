# D5-L2E — Attention Impact Computation Optimization

작성일: 2026-11-21. 개인 Ubuntu에서 로컬 CPU 작업만 수행했다. Candidate는 **`MASK_INDEX_REUSE_PER_LAYER_V1` 하나**다. Layer별 공통 Int64 index와 optional padding-mask 전처리만 재사용하며, L2C의 host batching 및 각 Window의 FP64 reduction을 유지한다.

실제 반복 작업 감소와 **통합 소스 Native CPU 회귀 245/245 PASS**를 확인했고, 30회 CPU 측정에서 L2C보다 낮은 중앙값을 관측했다. 다만 분산이 크므로 일반적인 속도 향상 판정은 유보한다. GPU 성능·GPU fidelity·공식 CPU Approval은 이번 작업의 검증 결과가 아니다.

## 1. Baseline 확인

| 항목 | 시작 시 확인 |
|---|---|
| Repository / Branch | `GP_semcache_lightperf` / `exp/d5-l1-l5-lightperf` |
| HEAD | `e0b2d37f67b958ba28995b3639d140139c0b7b49` |
| 직전 이력 | `5c00c44` L2D 구현, `cc3dac0` 및 `e0b2d37` L2D 진단 증거; 그 이전 L2C `53407e7` |
| Strict baseline | `4916836ce328e119f3a8d85c6c1afc7cf8d363e5` |
| Tracked dirty | 시작 시 없음 |
| 기존 Untracked | `.venv-d5l2e/`, `d5_l2b_inputs.tar.gz`, `results/r5/` — 기존 항목 보존 |
| 기본 Python | `/usr/bin/python3`, 3.10.12; torch/transformers/numpy/pytest 없음, PyYAML 5.4.1 |
| 실제 사용 환경 | 기존 `.venv-d5l2e/bin/python`, Python 3.10.12 |
| Native versions | torch **2.14.1+cpu**, transformers **4.57.6**, numpy **2.2.6**, pytest 9.2.0, PyYAML 6.0.4 |
| CPU 확인 | 실제 FP64 Tensor 연산 및 Tiny OPT import/실행 성공. `torch.version.cuda is None`, CUDA 미초기화 |
| 설치 | **미실행**. 기존 사용자 준비 venv가 있어 설치할 필요가 없었음 |

CPU 명령에 `CUDA_VISIBLE_DEVICES=''`, `HIP_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, offline Hub/Transformers 환경을 적용했다. Native UD 검증의 localhost TCP는 sandbox 권한을 받아 실행했으며 원격 네트워크나 SERAPH 접속은 하지 않았다. GPU, Slurm 제출, commit, push, 실험 PLAN/Approval 생성은 실행하지 않았다. 테스트의 임시 TEST_FIXTURE_ONLY 문서는 실제 GPU PLAN이나 승인으로 취급하지 않는다.

기존 L2C는 `REFERENCE_WINDOW_SCALAR`를 default로 유지하고 `DEFERRED_HOST_PER_LAYER_V1`를 명시적 후보로 제공한다. L2D는 owner 진단만 추가하고 최적화 후보를 만들지 않은 상태다. 이번 시작 소스의 관련 회귀 **231개가 실제 Native CPU에서 전부 PASS**했다([baseline_native.log](../d5_l2e_diagnostics/baseline_native.log)). 이전 L2D의 13 FAIL/8 SKIP도 이번 환경에서는 재실행되어 통과했다. 이전 승인이나 이전 PASS를 새 소스 승인으로 재사용하지 않았다.

사용자 제공 D5-L3 Attention Impact 평균은 2WikiMQA 3.027 s/request, Qasper 2.989, MultiNews 2.979다. 이는 50 Mbps, UD100, Continuous Replay 20-request의 기존 GPU 실측이며 이번 CPU 측정과 구분한다. 해당 원본 결과는 수정하지 않았다.

## 2. Attention Impact 연산 구조

관련 소스:

- [PaperRowL2SumReducer](../src/semcache/cache/attention_impact.py), `reduce`: FP64 query slice → causal/padding mask → masked_fill → finite → norm(last axis) → mean(head) → sum(window) → scalar materialization.
- [StreamingImpact](../src/semcache/edgelora/r5_subsequence_generation.py), `observe`: layer당 실제 FP64 변환 한 번, window별 기존 reducer, Python layer accumulation 및 record append.
- [L2C](../src/semcache/edgelora/r5_d5_impact.py), `deferred_window_values`: 동일 연산을 window마다 수행하고 finite/scalar를 묶어 `.cpu().tolist()`를 layer당 한 번 실행.
- [L2E](../src/semcache/edgelora/r5_d5_impact_compute.py), `reused_window_values`: index/padding operands만 layer-local로 재사용.
- [Profiler](../src/semcache/edgelora/r5_subsequence_profile.py), [Light timing](../src/semcache/edgelora/r5_d5_lightperf_timing.py): `impact_fp64_conversion`, `actual_attention_impact_reduction` 등 기존 scope와 record 형식을 유지.

`N=512`, `L=32`, sliding `w=3`이면 **W=N-w+1=510**이다. head 수는 별도 변수 H이며 아래 bytes 예시는 실제 CPU benchmark에 맞춘 **H=32**다. 호출 수만으로 latency 병목을 단정하지 않는다.

| 작업 | Reference / L2C: layer당 | L2E: layer당 | Reference / L2C: 32 layers | L2E: 32 layers | 할당/동기화 의미 |
|---|---:|---:|---:|---:|---|
| 실제 FP16/FP32→FP64 cast | 1 | 1 | 32 | 32 | full attention FP64 storage 생성. 입력이 이미 FP64면 no-op 가능 |
| window 안 `detach().double()` API | 510 | 510 | 16,320 | 16,320 | 실제 streaming에서는 이미 FP64. 재호출과 실제 full-tensor 복사를 구분 |
| window attention slice | 510 | 510 | 16,320 | 16,320 | view; query 위치/순서 동일 |
| key `arange(N)` | 510 | 1 | 16,320 | 32 | Int64 device allocation |
| query `arange(start,end)` | 510 | 0 | 16,320 | 0 | L2E는 공통 index의 read-only slice |
| index 생성 합계 | **1,020** | **1** | **32,640** | **32** | query/key 값과 dtype/device는 동일 |
| optional padding `.to(bool)` 및 shape validation | 510 | 1 | 16,320 | 32 | 실제 dtype/device가 이미 같으면 `.to` 자체는 no-op 가능 |
| causal comparison `[w,N]` | 510 | 510 | 16,320 | 16,320 | 새로운 bool mask. 전체 NxN mask 없음 |
| optional padding outer AND / valid in-place AND | 각각 510 | 각각 510 | 각각 16,320 | 각각 16,320 | shared padding/index를 수정하지 않음 |
| `masked_fill` | 510 | 510 | 16,320 | 16,320 | FP64 `[H,w,N]` 새 tensor |
| `isfinite` 및 `.all()` | 각각 510 | 각각 510 | 각각 16,320 | 각각 16,320 | finite tensor/flag 생성; L2C/L2E는 flag를 device에 유지 |
| norm / mean / sum | 각각 510 | 각각 510 | 각각 16,320 | 각각 16,320 | 기존 axis/shape/순서 유지 |
| host reads | Reference: bool 510 + item 510; L2C: packed 1 | packed 1 | Reference 최대 32,640; L2C 32 | 32 | CPU 호출 횟수이며 GPU synchronization 시간과 같지 않음 |

CUDA에서 Reference의 Tensor bool, `.item()`은 device scalar 관측 경계다. L2C/L2E의 `.cpu().tolist()`는 layer당 한 번의 host materialization이다. `arange`, comparison, casts, masked_fill, finite 및 reduction은 CUDA kernel/allocation work가 될 수 있지만 그 자체를 모두 explicit synchronization으로 세지 않는다. allocator 등의 내부 동기화 가능성은 GPU 미측정이며 이번 작업에서 새 `.item()`, Tensor bool, `.synchronize()`를 추가하지 않았다.

현재 Light streaming의 `observe()`는 helper에 optional padding mask를 전달하지 않는다. 따라서 padding 전처리 재사용의 이득을 기존 D5-L3의 실제 mask 비용 절감으로 주장하지 않는다. helper에 CPU padding mask와 CUDA attention을 함께 주는 경우 `.to(device)`가 전송/대기를 포함할 수 있지만 그 경로도 이번에 GPU 측정하지 않았다. 본 작업에서 mask가 있는 helper 경로와 현재 mask 없는 Light 경로를 분리해 비교한다.

### 정적 메모리와 allocation traffic

| Tensor / 작업 | N512 / H32 예시 |
|---|---:|
| full attention probability | FP16 16 MiB / FP32 32 MiB / FP64 64 MiB |
| FP64 layer copy | 64 MiB, 필요 시 layer당 한 번 |
| key index | 512×8 = 4,096 bytes |
| query index | 3×8 = 24 bytes/window, L2E는 view |
| converted padding mask | 512 bytes, 원본 mask storage는 별도 |
| causal valid mask / inverse / padding outer mask | 각각 3×512 = 1,536 bytes (각 tensor가 생성될 때) |
| masked FP64 attention slice | 32×3×512×8 = 393,216 bytes |
| finite bool output | 49,152 bytes; isfinite 내부 scratch는 별도 |
| norm output / head mean / scalar | 768 / 24 / 8 bytes |
| L2C/L2E retained window outputs | 510 finite bool scalars 및 510 FP64 scalars, packing temporaries와 Python/allocator overhead 별도 |
| hypothetical full NxN bool causal mask | 262,144 bytes/layer, head broadcast materialization은 별도. 채택하지 않음 |

기존 index의 누적 payload allocation은 layer당 `510×(512+3)×8=2,101,200 bytes`, 32 layers에 **67,238,400 bytes**다. L2E는 layer당 4,096 bytes, 전체 **131,072 bytes**다. 차이 **67,107,328 bytes**는 누적 allocation traffic이며 peak memory 감소 수치가 아니다. L2E의 index는 layer 끝까지 살아 있으므로 순간 peak는 오히려 조금 증가할 수 있다.

`masked_fill`의 FP64 결과만 따져도 누적 payload는 layer당 200,540,160 bytes이며, 이 작업은 줄이지 않았다. `isfinite`는 backend에서 abs/비교 등의 추가 scratch를 만들 수 있으므로 단순 output bytes 합이 실제 peak가 아니다. 실제 CPU tensor allocator timeline을 별도 측정한다(5절). GPU allocator reserve/RSS로 혼동하지 않는다.

Resource bound에는 기존 4 MiB staging reserve 외에 `ceil(9N/4096)×4096` bytes를 추가한다. 정상 Light의 N≤512에서 **8,192 bytes**이며 Int64 index 8N 및 optional bool copy N을 보수적으로 포함한다. PLAN의 `impact_layer_working_bytes`, `gpu_estimate_bytes` 양쪽에 반영하고 runtime은 실제 N으로 계산한다. 기존 128 MiB probability, 1 GiB working set, 22 GiB ceiling 등의 상한을 늘리지 않았다.

## 3. 최적화 Candidate

### 선택과 변경 범위

**`MASK_INDEX_REUSE_PER_LAYER_V1`**는 Candidate A와 B의 operand 재사용을 하나의 후보로 묶는다. 함수 호출/layer를 넘는 캐시나 검증 결과 memoization은 없다.

Before:

```python
valid = arange(N)[None, :] <= arange(start, end)[:, None]
# optional: window마다 .to(bool), 1D/2D shape 처리 및 validation
```

After:

```python
# 첫 유효 window의 기존 attention/bounds 검사 뒤 한 번
positions = arange(N, device=values.device)
# optional: mask를 한 번 .to(device, bool)하고 기존 shape 검사
valid = positions[None, :] <= positions[start:end, None]
```

각 window는 계속 같은 `[start:end]` query rows와 모든 causal valid keys를 사용한다. Causal mask와 padding outer AND는 새 tensor를 만들어 결합하며 cached index/mask 및 attention 원본은 변경하지 않는다. 일반 PyTorch 연산과 마찬가지로 호출 중 입력을 외부에서 동시에 변조하는 사용은 지원하지 않는다.

첫 window에서 lazy 초기화하여 empty list의 기존 조기 반환, shape/bounds 오류 우선순위, 1D/단일행 2D mask 검증을 유지했다. `masked_fill`로 causal/padding 밖 NaN/Inf를 0으로 바꾸고 valid cell의 nonfinite는 기존 finite flags로 검출한다. 이후 norm/mean/sum, packed host read, Python `0.0 + value`, layer accumulation, dictionary/record 순서도 유지한다.

L2C에서 이미 바뀐 batched failure 경계를 유지한다. 일부 window scalar가 device에서 계산되었더라도 finite/overflow 오류가 있으면 반환하지 않는다. streaming layer의 values/record/observed/host-call 성공 카운터를 갱신하기 전에 실패한다. Native 모델 hook 실패는 기존 owner.abort/projection-hook restoration/caller cleanup으로 전달되며 Admission으로 진행하지 않는다.

### 채택하지 않은 변경

- C: 공통 index로 comparison operand만 재사용했다. full NxN mask를 추가하거나 causal 비교 횟수를 바꾸지 않았다.
- D: FP64 cast 위치/Norm/Mean/Sum/finite/masked_fill을 합치거나 바꾸지 않았다. 이미 FP64인 `.double()` API 호출을 남겨 기존 helper의 direct FP16/32 사용까지 계산 의미를 보존했다.
- Token-level norm precompute, prefix sum, window 재배열/제거, cross-layer reassociation, 근사 연산은 구현하지 않았다. 범위 밖 셀의 nonfinite를 잘못 검증하거나 floating reduction 순서를 바꿀 여지가 있다.

### LightPerf 통합과 승인 경계

통합 전에 별도 모듈에 대해 Native reduction 7개 및 실제 Tiny OPT gate 2개를 먼저 통과했다([reducer_parity.log](../d5_l2e_diagnostics/reducer_parity.log), [tiny_opt_gate.log](../d5_l2e_diagnostics/tiny_opt_gate.log)). 이때 임시 CPU test factory로 Candidate를 주입했으며 production selector에는 연결하지 않았다. 통합 후 테스트는 새 selector를 직접 사용한다.

| 통합 지점 | 처리 |
|---|---|
| selector/factory | 세 구현을 허용. default 및 누락 field는 계속 `REFERENCE_WINDOW_SCALAR`. 신규 값은 명시적으로 선택 |
| PLAN binding | 기존 plan hash에 implementation field 포함. prepare/load에서 신규 reserve를 재계산. selector만 바꿔 reseal한 잘못된 resource binding은 거부 |
| source fingerprint | 기존 `source_identity`가 신규 src/scripts/tests/docs를 포함. dirty/uncommitted 소스 및 기존 fingerprint로 clean approval을 만들 수 없음 |
| resource bounds | 새 후보에만 8 KiB(정상 N512) 추가. Reference와 L2C의 기존 resource 결과 보존 |
| dispatch/receipt | 기존 runtime→generate_native→create_impact 경로 사용. compact Impact receipt에 새 implementation label 및 layer당 host call 기록 |
| official verifier | 기존 generic implementation equality, 실제 32-layer Impact coverage/32 host calls 및 modules binding을 그대로 적용. 잘못된 label/host count/구 L2C 승인 목록을 거부하는 신규 테스트 추가 |
| CPU approval modules | 기존 L2C는 기존처럼 `test_r5_d5_impact`만 추가. L2E는 이에 `test_r5_d5_impact_compute`, `test_r5_d5_owner_diagnostics`를 추가 요구 |
| analysis/review | 기존 variant grouping 및 FULL/EXACT implementation mismatch 거부 유지. paired reviewer는 Reference→L2C, Reference→L2E, L2C→L2E를 구분하고 두 label을 결과에 명시 |

새 GPU PLAN, 공식 CPU Approval, reviewed commit을 생성하지 않았다. 새 selector 지원은 GPU 실행 허가와 다르다. 사용자 commit/review 후 새로운 clean source/fingerprint/PLAN/CPU Approval 및 별도 GPU 승인이 필요하다.

## 4. Native CPU Parity

최종 통합 소스에서 **245 PASS, 0 FAIL, 0 ERROR, 0 SKIP**이며 실행 exit code=0, 384.496s다. [final_native_regression.log](../d5_l2e_diagnostics/final_native_regression.log)에 보존했다.

| 범위 | 결과 |
|---|---|
| 시작 소스 기존 회귀 | 231/231 PASS, 0 FAIL/ERROR/SKIP |
| 통합 전 standalone gate | reduction 7/7 및 Tiny OPT 2/2 PASS |
| 통합 후 신규 모듈 | 14/14 PASS: native reducer 7, direct-selector Tiny OPT 2, integration contracts 5 |
| 통합 후 기존 회귀 포함 전체 | **245/245 PASS**, 기존 L2C/L2D/Native/무결성 테스트 유지 |
| CPU 연산/메모리 관측 | 30회×세 구현×세 조건 정상 완료, 실제 API 반복 감소 및 동일 FP64 hash |
| 일반화 가능한 CPU speedup | **판정 유보**: 중앙값 감소 관측, 큰 분산과 실행별 역전 존재 |
| GPU 성능/fidelity | **NOT_RUN / NOT_VERIFIED** |
| 공식 CPU Approval / GPU 승인 | **NOT_CREATED / NOT_APPROVED** |

최종 모듈 목록은 `test_r5_d5_impact_compute`, `test_r5_d5_impact`, `test_r5_d5_lightperf_native`, `test_r5_d5_owner_diagnostics`, `test_r5_subsequence_validation`, `test_r5_subsequence_generation`, `test_r5_sparse_physical`, `test_r5_natural_approx`, `test_r5_d5_lightperf`, `test_r5_d5_lightperf_worker`, `test_r5_d5_lightperf_replay`, `test_r5_natural_shutdown`, `test_r5_natural_phase3a`, `test_r5_subsequence_audit`, `test_r5_subsequence_diagnostics`다. 전체 repository의 모든 테스트를 실행했다는 뜻은 아니다. Native dependency 또는 Runtime Compatibility 오류로 남은 항목은 없으며 이 venv/테스트 범위에서의 결과다.

재현 시 기존 venv, CUDA/HIP hidden, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests`에서 `.venv-d5l2e/bin/python -m unittest <위 모듈들> -v`를 실행한다. localhost TCP 권한이 필요하다. 이 unittest 실행은 clean source와 PLAN에 binding된 공식 `cpu-validate`를 대신하지 않는다.

검증 코드: [test_r5_d5_impact_compute.py](../tests/test_r5_d5_impact_compute.py). Native 의존성이 없으면 실패하도록 했으며 새 skip decorator나 허용 오차를 추가하지 않았다.

- FP16/FP32/FP64, heads 1/3/8, N=1/2/3/6/17/64/512, sliding w=3, 양끝 width-1 boundary, 반복 token 및 반복/역순 window를 비교했다. 비연속 attention 및 padding mask, negative zero, bool/int64/float32의 1D/단일행 2D padding도 포함한다.
- 유효 window 값은 `struct.pack('<d', value)` bytes로 Reference/L2C/L2E를 비교한다. 실패 메시지에 implementation, layer, dtype, window bounds와 양쪽 FP64 `.hex()`를 기록한다. masked NaN/Inf는 입력 uint8 bytes 보존까지 확인하며 valid NaN/Inf·overflow·invalid shape/mask/window는 오류로 처리한다.
- 별도 streaming test는 3-layer values/records/order를 비교하고 실패 layer가 이전 성공 layer 결과를 바꾸지 않는지 확인한다. Python `1e16,1,1` 순차 누적 사례도 유지한다.
- Tensor `.item()`/`__bool__`를 실패로 바꾼 실행에서도 L2E는 성공하고, layer당 index 1회, optional source mask conversion 1회, packed CPU transfer 1회임을 실제 API 관측으로 확인한다.
- Tiny OPT는 실제 Torch CPU QKV/LoRA와 별도 localhost UD를 사용한다. 전체 vocabulary logits의 float32 bytes, Generated IDs/Text, QKV fresh/reuse rows, snapshots, HIT/MISS, selection, cache metadata/positions/F/I/timestamps, History, CHU/PBR/Admission/Eviction, occupancy, Entry proofs 및 integrity counters/authorizations를 세 구현에서 비교한다. Impact SHA256도 같아야 한다.
- 실제 Native cache 내용 변조, deadline, Impact hook 예외에서 failure/abort/cleanup을 검증한다. Impact 실패 시 owner.complete/Admission이 호출되면 실패하며 active/pending/validation scope가 정리되는지 확인한다.
- 기존 L2C/L2D 안전성·무결성·alias/storage 변조 테스트를 변경하지 않았다. Strict reducer, PhysicalOwner, RequestValidation, 기존 generation/profile 구현과 L2D 코드는 그대로다.

이 결과는 Tiny OPT/CPU 범위다. OPT 2.7B FP16/CUDA fidelity나 production 데이터셋 품질·GPU 성능을 승인하는 결과가 아니다.

## 5. CPU 성능 비교

최종 자료는 [native_cpu_benchmark.json](../d5_l2e_diagnostics/native_cpu_benchmark.json), 실행 로그는 [native_cpu_benchmark.log](../d5_l2e_diagnostics/native_cpu_benchmark.log)다. 세 구현의 모든 warmup/측정 결과에서 FP64 hash가 같았으며 프로세스는 정상 완료했다.

재현 harness는 [162_r5_d5_impact_compute_cpu.py](../scripts/162_r5_d5_impact_compute_cpu.py)다. N512/H32, FP32 probability, FP64 compute, CPU thread/inter-op thread=1, 동일 입력·mask를 사용한다. 각 조건에서 warmup 3회 후 구현 순서를 회전하며 30회 측정한다. 세 조건은 no padding 1 layer, padding 2D 1 layer, 실제 32-layer streaming 전체다. 32-layer 사례는 동일 attention tensor를 layer마다 재사용하는 합성 workload이며 모델/E2E 비용은 포함하지 않는다.

입력은 고정 seed의 random Tensor이며 실제 model softmax probability로 정규화한 데이터는 아니다. Native 연산 구조/shape의 CPU 비교로만 사용한다. 모든 primary 측정은 별도 테스트가 끝난 후 단일 benchmark 프로세스에서 수행했고, 최종 회귀는 benchmark 종료 후 시작했다.

주 timing은 observer/torch profiler 없이 수행한다. bit hash 비교는 timer 밖이다. 별도로 API wrapper 30회 관측과 CPU-only torch memory profiler를 실행해 관측 비용·operator count·tensor allocator peak를 분리한다. `amortized_window_ms`는 전체 layer 시간을 W로 나눈 값이며 순수 norm kernel 시간이라고 하지 않는다. CPU frequency/affinity/system-wide background는 고정하지 않았으며 GPU gain으로 환산하지 않는다.

### Uninstrumented 시간

아래는 **median [p25,p75], ms**, 각 30회다. 32-layer stream은 실제 32회를 실행한 전체값이며 single-layer 측정에 32를 곱한 추정치가 아니다.

| 조건 | Reference | L2C | L2E | L2C 대비 중앙값 감소 |
|---|---:|---:|---:|---:|
| N512/H32, 1 layer, mask 없음 | 351.797 [325.519,449.770] | 323.927 [300.050,441.601] | 302.229 [285.960,374.227] | 6.70% |
| N512/H32, 1 layer, 2D padding | 374.471 [345.362,493.190] | 369.935 [331.329,421.088] | 339.869 [307.491,409.031] | 8.13% |
| N512/H32, 32-layer stream, mask 없음 | 12,852.631 [12,691.410,13,360.633] | 12,296.554 [11,998.098,13,010.663] | 11,652.399 [11,354.736,12,584.672] | 5.24% |

Amortized window time(ms)은 같은 순서로 mask 없음 **0.68980 / 0.63515 / 0.59261**, padding **0.73426 / 0.72536 / 0.66641**, 32-layer stream **0.78754 / 0.75347 / 0.71400**다. layer cast/validation/host read 및 streaming bookkeeping을 포함하며 isolated reduction kernel 시간과 다르다.

각 repeat에서 `(L2C-L2E)/L2C`를 계산한 paired 관측:

| 조건 | paired median [p25,p75] | L2E가 빠른 반복 |
|---|---:|---:|
| mask 없음, 1 layer | 5.99% [-5.35%,11.93%] | 18/30 |
| padding, 1 layer | 4.42% [-3.66%,20.59%] | 19/30 |
| 32-layer stream | 3.83% [-2.51%,9.00%] | 21/30 |

IQR가 겹치고 paired 분포에도 역전이 있다. 32-layer L2C의 최대값은 31.532s, L2E는 18.753s이며 primary mean도 outlier 영향을 받는다. 결과를 삭제하거나 시간을 보정하지 않았다. **연산/할당 중복 감소는 확인했지만, 안정적인 CPU latency 개선 또는 일반화 가능한 speedup을 확정하지 않는다.** 보고하는 수치는 이 한 환경의 descriptive observation이다. 특히 이를 기존 GPU Impact 약 3초에 곱해 예상 절감량으로 제시하지 않는다.

### 실제 API census 및 작업별 진단

N512/H32 한 layer의 30회 별도 관측에서 모든 반복의 호출 수가 같았다. Reference와 L2C는 `arange=1,020`, L2E는 `1`; padding 변환은 mask가 있을 때 `510 → 1`이다. 세 버전 모두 comparison/masked_fill/isfinite/all/norm/mean/sum은 각각 **510회**, `double` API는 외부 cast 포함 **511회**다. L2C/L2E는 `.cpu` 및 `.tolist` 각각 **1회**, `.item` 및 Tensor bool **0회**였다. Reference는 item/bool 각각 510회다. Python API 호출 수이며 ATen 내부 호출/실제 kernel 수와 혼합하지 않는다.

별도 wrapper의 component inclusive median(ms), mask 없는 한 layer:

| 작업 | L2C | L2E | 해석 |
|---|---:|---:|---|
| arange 전체 | 19.955 | 0.072 | 실제 호출 감소. timer/wrapper overhead 포함 |
| causal comparison | 7.636 | 6.767 | 두 구현 모두 510회 |
| masked_fill | 127.678 | 86.378 | 횟수/수식 동일; 아래 차이를 해당 연산 최적화로 해석하지 않음 |
| isfinite | 242.531 | 175.718 | 동일 510회 |
| norm / mean / sum | 57.869 / 29.016 / 7.536 | 39.207 / 20.894 / 5.822 | 세 reduction의 shape/순서/횟수 동일 |

Padding `.to`의 별도 계측 median은 L2C 6.265ms(510회), L2E 0.032ms(1회)다. observer는 구현별 연속 cohort에서 실행했으므로 시점/시스템 분산이 있고, component 값에 Python timer·wrapper 비용이 들어간다. 변경하지 않은 finite/norm 등의 시간 차이를 새 계산 단축의 증거로 삼지 않는다. 독립 중앙값이나 중첩 component를 합산해 총시간을 재구성하지 않는다. JSON에는 CPU ATen operator self time 및 allocation metadata도 보존한다.

관측 wall median은 mask 없음에서 Reference/L2C/L2E **469.235 / 646.181 / 462.395ms**, padding에서는 **548.498 / 495.715 / 414.244ms**다. primary timing보다 상당히 크다. 설치/해제까지 포함한 다른 cohort의 중앙값 차이이므로 이를 정밀한 overhead 보정 상수로 빼지 않는다. 성능 비교는 위의 uninstrumented primary 값으로만 판단한다.

### CPU tensor allocator memory

`torch.profiler`의 CPU memory timeline에서 원래 input을 포함한 peak live tensor bytes와 CREATE 누적량을 별도로 계산했다. Python object/RSS/allocator bookkeeping/비-Tensor workspace 및 GPU VRAM 측정이 아니다. 아래 profile은 32-layer 전체가 아닌 **한 layer**다.

| 조건 | 구현 | peak live bytes | preexisting bytes | CREATE 누적 bytes |
|---|---|---:|---:|---:|
| mask 없음 | Reference | 101,598,720 | 33,554,432 | 547,480,414 |
| mask 없음 | L2C | 101,603,301 | 33,554,432 | 547,497,244 |
| mask 없음 | L2E | 101,607,397 | 33,554,432 | 545,400,140 |
| padding | Reference | 101,601,280 | 33,556,480 | 548,524,894 |
| padding | L2C | 101,605,861 | 33,556,480 | 548,541,724 |
| padding | L2E | 101,609,957 | 33,556,480 | 546,184,012 |

L2C 대비 CREATE 누적량은 mask 없음 **2,097,104 bytes**, padding **2,357,712 bytes** 줄었다. 후자는 index 차이와 `(510-1)×512` bool mask bytes 차이가 일치한다. 반면 persistent index 때문에 실제 CPU tensor peak는 두 경우 모두 **4,096 bytes 증가**했다. 새 reserve 8,192 bytes 안에 있지만 이는 GPU allocator peak 검증을 대신하지 않는다. FP32 CPU 입력의 runtime working estimate는 105,259,008 bytes이며 이번 tensor peak보다 크다. 실제 FP16 production GPU 입력의 상한 계산/실측은 별도다.

재현 명령(기존 venv 사용; fresh output 필요):

```bash
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src:tests \
  .venv-d5l2e/bin/python scripts/162_r5_d5_impact_compute_cpu.py \
  --output /tmp/d5_l2e_cpu_benchmark_new.json \
  --tokens 512 --heads 32 --layers 32 --repeats 30
```

이 설정은 실제 30×32-layer×세 구현을 실행해 CPU 시간이 길다. 소형 `--tokens 6 --heads 2 --layers 2`는 harness 점검용이며 위 수치의 대체 검증이 아니다.

동일 소스의 소형 `benchmark_smoke.json`은 harness/memory timeline 점검용으로만 보존하며 N512 성능 결과에 합치지 않는다. 진단 output은 raw attention/logits/입력 덤프 대신 timing, count, allocation metadata, bit hash와 source hashes만 저장한다.

## 6. 변경 파일

| 파일 | 목적 |
|---|---|
| 신규 `src/semcache/edgelora/r5_d5_impact_compute.py` | Layer-local index/padding 재사용 helper, 별도 streaming class, 추가 reserve 계산 |
| `src/semcache/edgelora/r5_d5_impact.py` | 명시적 selector/factory 및 추가 resource binding. **기존 L2C helper/class 본문 byte-identical** |
| `src/semcache/edgelora/r5_d5_lightperf.py` | prepare/load resource 적용 및 신규 후보 CPU module 요구. 기존 defaults/L2C 조건 보존 |
| `src/semcache/edgelora/r5_d5_impact_review.py` | 세 버전의 명시적 paired comparison 및 labels |
| `scripts/160_r5_d5_impact_compare.py` | L2E 비교 지원에 맞춘 help docstring |
| 신규 `tests/test_r5_d5_impact_compute.py` | Native bitwise/Tiny OPT/failure 및 PLAN/verifier/approval/analysis 계약 테스트 |
| 신규 `scripts/162_r5_d5_impact_compute_cpu.py` | 30회 uninstrumented CPU benchmark, 별도 API census 및 tensor memory profile |
| 신규 `docs/D5_L2E_ATTENTION_COMPUTE_OPTIMIZATION.md` | 본 보고서 |
| 신규 `d5_l2e_diagnostics/` | Native 로그 및 benchmark/provenance. 기존 L2C/L2D/L3 결과와 분리 |

`attention_impact.py`, `r5_subsequence_generation.py`, `r5_subsequence_profile.py`, `r5_subsequence_physical.py`, `r5_subsequence_integrity.py`, L2D 파일은 시작 HEAD와 byte-identical이다. L2C `deferred_window_values`, `DeferredHostImpact`의 함수/class source도 AST source-segment로 정확히 비교했다. 기존 receipt/timing/runtime/analysis generic 경로는 신규 selector를 전달할 수 있어 해당 파일의 schema나 safety guard를 수정하지 않았다.

현재 전체 source inventory의 [development_source.json](../d5_l2e_diagnostics/development_source.json) 및 [local_verification.json](../d5_l2e_diagnostics/local_verification.json)에 파일 hash와 로컬 검증 provenance를 보존한다. development fingerprint의 scope는 `DEVELOPMENT_NOT_GPU_APPROVED`다. 새 Reviewed HEAD는 **사용자 commit/review 전이므로 미생성**이며 현재 dirty source의 fingerprint를 기존 clean approval에 대입하지 않는다.

## 7. 후속 GPU 실험 계획

1. 사용자가 최종 변경을 review/commit한 **새 HEAD와 content fingerprint**가 필요하다. 현재 HEAD는 시작 commit이고 worktree는 새 코드로 dirty하므로 `CLEAN_REVIEWED_SOURCE`가 아니다. 개발 fingerprint는 검토 자료일 뿐 승인으로 쓰지 않는다.
2. 새 clean source에 대해 Reference/L2C/L2E 각각 fresh PLAN 및 Native CPU Approval을 binding한다. 모든 요구 모듈은 zero FAIL/ERROR/SKIP이어야 한다. 여기의 unittest/benchmark 로그는 공식 CPU approval을 대신하지 않는다. 기존 L3 승인/PLAN/receipt를 수정하거나 재사용하지 않는다.
3. 별도 사용자 GPU 승인 뒤 2WikiMQA부터 시작한다. 동일 RTX 3090/UUID/backend/precision/CPU affinity/seed/task+stream hash, **50 Mbps, UD100, Continuous Replay 20 requests**, EXACT+SCOPED를 유지한다. L2C→L2E paired EXACT가 일차 비교이며 Reference도 필요한 보조 기준으로 둔다.
4. 각 실행은 condition별 fresh owner/session/cache로 시작하고 20 requests 내에는 cache lifetime을 유지한다. 동등한 관측 수준을 적용하고 가능한 경우 순서를 교차한 반복 trial을 사용한다. 추가 GPU synchronize를 숨기지 않는다. 새 임시 index/mask와 기존 reserve가 실제 VRAM 상한 안에 있는지 확인한다.
5. Impact FP64/hash, host calls, QKV rows, selection, CHU/PBR/Admission/Eviction, occupancy, proofs/failures, Generated IDs/Text/품질을 확인한 후 Attention Impact elapsed, TTFT, E2E를 비교한다. Light에 없는 dense QKV/full-vocabulary fidelity는 별도 검증 없이 PASS라 하지 않는다.
6. 연구 목표의 FULL vs EXACT를 위해 새 source에서 동일 50 Mbps/UD100/replay20의 matched FULL도 측정한다. implementation label이 다른 FULL/EXACT를 기존 analysis에 섞지 않는다. 기존 L3 수치는 역사적 참고로 두고 새 source의 matched comparison을 주 결과로 삼는다.
7. 2WikiMQA의 보존 및 자원 조건이 통과한 뒤 Qasper/MultiNews로 확장한다. CPU에서 확인한 index/mask 호출 감소와 시간 차이를 GPU 성능 향상의 증거로 사용하지 않는다.

이번 작업에서는 GPU 실행, Slurm 제출, SERAPH 접속, 설치, 공식 승인 생성, commit/push를 하지 않는다. 후속 GPU 실행은 이 보고서가 자동으로 승인하지 않는다.
