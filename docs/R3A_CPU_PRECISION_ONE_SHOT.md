# R3-A CPU LoRA precision: one allocation, three candidates

분류는 `R3A_PARITY_DIAGNOSTIC`, `PRIMARY_PERFORMANCE_ELIGIBLE=false`다.
CPU-only UD에 적용할 수 있는 산술 후보를 **로컬 CPU delta 주입 방식**으로
검사한다. 생산 ES/UD 경로, 기존 95/96/97, 임계값, 과거 자료는 수정하지 않는다.
이번 실행은 TCP, pilot, 벤치마크, cache reuse, codec, quantization을 수행하지
않는다. 후보가 PASS해도 실제 CPU-UD/TCP 경로가 검증된 것은 아니다.

## 이전 관측과 이번 cohort

사용자가 제공한 SERAPH job151730 관측은 다음과 같다. 개인 Ubuntu의 Codex가
원본 파일을 열거나 독립 검증한 자료가 아니다.

- moana-y4 RTX3090, 새 reference와 native control은 정확히 재현됨.
- physical pilot, local CPU oracle, TCP replay 모두 native full-logit strict FAIL.
- oracle/replay 전체 FP32 logits SHA는
  `5fd13d7fe978c690b52a5410fc273fe66783c7ba7a74c2d62c13dbca0e8cb284`,
  QKV 출력 96개 SHA도 동일하다고 보고됨.
- max_abs=0.0546875, relative_l2=0.0019029290937084956,
  tolerance_violations=828548, KL=1.1583049624620668e-06, argmax 일치.
- 97의 공식 완료는 `torch_config_sha256` 메타데이터 gate에서 막혔다.
  이 문서는 해당 결과를 공식 causal PASS로 소급 변경하지 않는다.

98은 과거 prepared/pilot/forensic 입력 옵션을 제공하지 않는다. 실제 새
allocation의 GPU로 95 prepare를 실행해 **새 immutable reference와 adapter**를
만든다. 과거 job의 UUID/숫자와 달라도 새 cohort 안의 UUID·identity 일치가
유효하면 진행한다. 원본 151730의 자동 로드/검증 flag는 항상 false다.

## 고정된 세 후보와 실제 연산

모든 입력은 동일 fresh cohort의 token IDs, bare OPT-2.7B FP16, batch1 L32,
eager attention, use_cache=False, output_attentions=True, all-token logits다.
32 layers의 Q/K/V, user_a rank=alpha=8, dropout=0, scale=1.0이며 원본 A/B는
CPU FP32로 보존한다. incoming hidden은 GPU FP16에서 CPU로 정확히 복사된다.

| 후보 | CPU A/B 계산 | 반환 delta |
|---|---|---|
| CPU_FP32_BASELINE | 기존 `user_device.cpu_deltas(..., torch_backend=True)` 직접 호출; FP32 A → FP32 B → ×1.0 | FP32 |
| CPU_FP64_FULL | hidden의 FP32 값과 원본 FP32 A/B를 FP64로 승격; FP64 A → FP64 B → ×1.0 | 최종 FP32 cast |
| CPU_FP64_STAGED | FP64 A → 중간 FP32 반올림 → 다시 FP64 → FP64 B → ×1.0 | 최종 FP32 cast |

세 후보의 A/B 연산은 전부 CPU에서 수행한다. GPU는 bias를 포함한 native
bare Linear와 `(FP16_base + returned_FP32_delta).to(FP16)`만 실행한다.
GPU LoRA A/B는 prepare/native control에서만 허용한다. FP64가 native GPU
FP32에 더 가깝다고 전제하지 않는다. 정확도와 GPU FP32의 반올림 재현은 다르다.
이 단계에서 별도 UD 프로세스/TCP를 사용하지 않는 것은 명시적 진단 범위다.
생산 CPU-only UD 구조에는 어떤 변경도 가하지 않는다.

## 실행 순서와 gate

98 parent에는 모델을 올리지 않는다. 이전 child의 종료와 cleanup을 확인한
후에만 다음 child를 실행한다. GPU에는 한 모델 사본만 존재한다.

1. Allocation identity child: 실제 `cuda:0` UUID와 버전·CPU affinity를 기록하고
   종료한다. CUDA context만 확인하며 model inference는 하지 않는다.
2. 기존 95 prepare CLI를 별도 subprocess로 실행한다:

   ```bash
   python -u scripts/95_run_r3a_distributed_no_reuse.py \
     --phase prepare --model facebook/opt-2.7b --dtype float16 \
     --prompt-lengths 32 --device cuda:0 --adapter user_a --seed 42 \
     --output-dir <NEW_RUN_ROOT>/prepare
   ```

3. Independent native control child: exported A/B를 정확히 재로드한 PEFT의
   forward **1회**. Native full-logit SHA와 strict metrics, 96 hidden/QKV/delta
   SHA를 fresh reference와 확인한다. Passive hooks는 출력/가중치를 바꾸지
   않는다. 같은 native hidden에서 세 CPU 후보를 모두 계산하여 projection
   delta 오차·최종 FP16 flip count를 기록한다. 추가 모델 forward는 없다.
4. 세 후보 각각 별도 child에서 bare-OPT full forward **1회씩** 실행한다.
   각 실제 layer당 CPU bundle을 한 번 계산하고 q/k/v가 같은 hidden을
   사용하는지 확인한다. 후보의 propagated hidden과 실제 QKV를 기록한다.
   Baseline layer0 bundle은 frozen CPU 함수의 독립 재호출과 bitwise 비교한다.

총 model forward는 prepare 1 + native 1 + 후보 3 = **5회**다. Warmup, 반복
latency, 새 pilot/forensic/97, 자동 추가 후보/파라미터 sweep은 없다.
원래 prepare가 수행하는 projection decomposition checks는 그대로 유지한다.

각 단계는 원본 source map, prepared artifact SHA, token IDs/hash,
model/tokenizer snapshot, 전체 base fingerprint, 실제 A/B dtype/shape/SHA,
versions와 새 GPU UUID를 확인한다. 기존 `validate_prepared()`를 그대로 쓴다.
프로그램/launcher 및 재사용한 diagnostic helper의 SHA도 별도로 기록한다.
다른 GPU 노드를 허용하지만 **동일 실행 안에서 GPU가 바뀌면 fail closed**다.
실제 CPU thread/affinity는 allocation에서 정해 모든 후보에 동일하게 적용한다.
Torch build-config SHA는 telemetry로만 기록한다. 기존 97 gate를 고치거나
우회하여 과거 결과를 승격시키는 작업은 하지 않는다.
95가 생성한 immutable prepare 파일은 기존 95 schema/classification을 그대로
보존한다. 새 study 보고서와 worker/JSONL 관측은 모두 진단 분류다.

native control이 불일치하면 후보는 NOT_RUN으로 남기고 FAILED로 종료한다.
예외/timeout은 traceback, stdout/stderr, lifecycle과 부분 trace를 남긴다.
prepare95가 별도 process group에 생성한 GPU child도 추적·정리한다.
완료되지 않은 후보의 parity를 실패/성공으로 채워 넣지 않는다.

## 수치 비교를 읽는 법

각 후보의 전체 `[1,32,vocab]` logits를 기존 `check_parity()`와 frozen
THRESHOLDS로 검사한다. max_abs/relative_l2/violations/KL/argmax와
전체 little-endian contiguous FP32 SHA, bytes, shape, 원래 logits dtype를
함께 보존한다. 기본적으로 큰 raw tensor dump는 저장하지 않는다.

96개 실제 QKV SHA, hidden/base/delta SHA와 첫 hidden/QKV 차이 layer/role도
기록한다. `matched_input_comparison.status`가
`MATCHED_INPUT_MEASURED_AND_HASH_BOUND`인 행만 실제 동일 입력 수치 비교다.
이는 native worker에서 측정한 같은-hidden 후보 결과를 실제 candidate의
hidden/base/delta/QKV SHA에 모두 연결한 것이다. Hidden만 같고 나머지가
다르면 `MATCHED_INPUT_PROBE_BINDING_MISMATCH`로 실패한다.

`NOT_APPLICABLE_CHANGED_HIDDEN`인 이후 행에는 actual matched-input 오차와
flip count를 null로 남긴다. 그 행의 `separate_native_hidden_counterfactual`은
**native의 원래 hidden**에서 계산한 별도 실측 control이다. 실제 candidate의
달라진 hidden에 대한 오차로 해석하지 않는다. SHA 불일치에 오차 크기를
추정해 붙이지 않는다. FP16 flip count는 수치적으로 달라진 원소 수이며,
bitwise 차이는 별도 component metric으로 보존된다.

후보 사이의 “가까움”은 단일 임의 점수로 순위를 만들지 않는다. 전체 지표를
비교하고 최종 합격 여부는 모든 기존 strict 조건을 만족해야 한다.

- 세 후보 모두 FAIL: `NO_CPU_PRECISION_CANDIDATE_PASSED`. 분석 자체는
  COMPLETE이며 negative result를 보존하고 여기서 종료한다.
- 하나라도 PASS: `CANDIDATE_LOCAL_PARITY_PASS`와 후보명을 기록한다.
  별도 후속 작업에서 실제 CPU UD/TCP 경로에 적용해 strict parity를 검증해야
  한다. 자동 생산 적용, pilot 승격, 성능 주장 또는 임계값 변경은 없다.
- 이 특정 CPU 수치/FP16 endpoint 검사가 실패해도 그것만으로 M1–M9,
  C1.5–C9 전체 연구가 실패한 것은 아니다. 각 단계의 실제 evidence scope와
  검증 주장을 유지하면서 R3-A 물리적 정확성의 현재 한계로 보고한다.

## SERAPH 동기화 및 단일 제출

추가 파일은 `scripts/98_explore_r3a_cpu_precision.py`,
`src/semcache/edgelora/cpu_precision_study.py`,
`tests/test_cpu_precision_study.py`, 이 문서,
`jobs/r3a_cpu_precision_one_shot.sbatch`다. 기존 파일은 변경하지 않는다.
로컬에서 검토·커밋하고 사용자의 기존 방식으로 원격 동기화한 뒤 SERAPH에서:

```bash
cd /data/khuss/repos/GP_semcache
git switch exp/raw-semcache-r3
git pull --ff-only
sbatch jobs/r3a_cpu_precision_one_shot.sbatch
```

Launcher는 `batch_eebme_ugrad`, GPU1, CPU4, RAM64G, 2시간 상한이며 특정 node를
지정하지 않는다. 시간은 자원 상한이며 예상 실행 시간 주장이 아니다.
`conda activate semcache`, `HF_HOME=/data/khuss/huggingface`,
`HF_HUB_CACHE=$HF_HOME/hub`, offline mode를 설정하고 allocation의
CUDA_VISIBLE_DEVICES를 보존한다. 실제 allocated UUID는 98의 identity child가
출력한다. 원본과 같은 UUID를 찾아 재할당할 필요는 없다.

기본 repository 경로는 `/data/khuss/repos/GP_semcache`다.
다르면 `R3A_PRECISION_REPO`를 지정한다. Batch 환경에서 conda 명령이 PATH에
없으면 실제 설치 경로를 `R3A_CONDA_ROOT`로 지정한다. Launcher 실행 전
동일 semcache 환경에 Torch/NumPy/Transformers/PEFT와 로컬 pinned 모델이
준비되어 있어야 한다. 다운로드나 dependency 자동 설치를 하지 않는다.

Run root는 `results/r3/a/precision_study_<jobid>_<UTC timestamp>/`이며 이미
존재하는 nonempty 경로는 덮어쓰지 않는다. Slurm 로그는
`/data/khuss/r3a_cpu_precision_<jobid>.out/.err`에 기록된다.
Codex는 sbatch를 제출하지 않았고 로컬 GPU 결과는 **NOT_RUN**이다.

## 결과 전달과 로컬 확인

다음 자료를 보내면 후보별 해석이 가능하다:

- `r3a_cpu_precision_summary.txt`와 `r3a_cpu_precision_results.json`.
- JSON의 `execution_status`, `gpu_findings_status`, `decision`,
  `prepared`, `allocation`, `native_control.native_exact`와
  `native_control.parity/logits/accounting`, 후보별
  `parity/logits/accounting/baseline_component_checks/process`.
- 실패했다면 `reason`, `traceback`, `worker_failures`와 해당 child의
  `stderr.log`, `lifecycle.json`, `native_mismatch.json`(있을 경우).
- 상세 projection 분석에는 `native/native_projection_probes.jsonl`과 각
  후보의 `candidate_projection_trace.jsonl`을 함께 전달한다.

`native_exact`가 true이고 후보별 지표가 모두 보이는지 먼저 확인한다.
`original_151730_evidence_loaded/verified`는 false여야 한다.

로컬 검증 명령:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_cpu_precision_study.py -v
python3 -m compileall -q scripts/98_explore_r3a_cpu_precision.py src/semcache/edgelora/cpu_precision_study.py tests/test_cpu_precision_study.py
python3 scripts/98_explore_r3a_cpu_precision.py --help
bash -n jobs/r3a_cpu_precision_one_shot.sbatch
```

Torch가 없는 개인 Ubuntu에서는 scalar TEST_FIXTURE_ONLY backend가 실제
frozen 함수의 위임과 두 FP64 후보의 연산 순서·반올림·불변성·실패 gate를
검사한다. 이것은 CPU BLAS/GEMM이나 GPU 수치 실험의 대체 증거가 아니다.
실제 Torch CPU 테스트 2개는 Torch가 있을 때만 모델 없이 실행한다.
GPU 테스트는 `SEMCACHE_R3A_PRECISION_GPU=1`과 새 경로인
`SEMCACHE_R3A_PRECISION_OUTPUT`을 명시한 GPU allocation에서만 실행한다.
개인 Ubuntu에서 해당 GPU opt-in을 활성화하지 않는다.

로컬 검증 결과: 신규 suite는 **19개 중 16 통과, 0 실패, 3 skip**이다.
실제 Torch CPU GEMM 2개와 GPU 실험 1개가 skip됐다. 관련 R3-A/forensic/
causal/R2-D/M9-A/C3/C4 회귀를 합친 **144개 중 138 통과, 0 실패, 6 skip**
이며, 나머지 skip도 기존 optional Torch/GPU 검사다. Compileall, CLI help,
`bash -n`이 통과했다. 기존 tracked 파일 507개는 SHA256으로 모두 불변임을
확인했다. GPU 실험 결과는 **NOT_RUN**이며, scalar CPU 테스트를 실제 CPU
BLAS 또는 GPU 정확도 결과로 승격하지 않는다.
