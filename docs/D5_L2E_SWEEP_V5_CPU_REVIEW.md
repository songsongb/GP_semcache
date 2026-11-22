# D5-L2E Bandwidth × UD Throughput Sweep v5 — CPU-first review

작업 범위는 **Phase A/B: 코드·로컬 CPU 테스트 및 소스 검토 자료 준비**다. 이 단계에서 멈춘다. 새 PLAN/CPU Approval/Slurm job/GPU 실행/commit/push를 생성하지 않는다. 아래 실행 예시는 사용자 소스 검토·commit/fingerprint 승인 **이후**를 위한 사용법이며 실행하지 않았다.

## 1. Baseline과 실제 수정 범위

- 로컬 저장소: `/home/songsong/GP_semcache_lightperf`
- 대상 SERAPH 저장소: `/data/khuss/repos/GP_semcache_lightperf` — **로컬에 없음**. 원격 접속하지 않음.
- Branch: `exp/d5-l1-l5-lightperf`, 시작 HEAD: `88af39e` (D5-L2E).
- 시작 Tracked 변경: 없음. 기존 Untracked `.venv-d5l2e/`, `d5_l2b_inputs.tar.gz`, `results/r5/` 보존.
- 사용한 기존 venv: Python 3.10.12, Torch 2.14.1+cpu, Transformers 4.57.6, NumPy 2.2.6, PyYAML 6.0.4. 설치/다운로드하지 않음.

| 파일 | 변경 목적 |
|---|---|
| `src/semcache/edgelora/r5_d5_lightperf_workload.py` | 기존 허용값에 `.10`만 추가. Generic matrix는 unlimited 포함 16 pairs/432 planning conditions. 기존 calibration/forecast/time/memory guard 유지 |
| `src/semcache/edgelora/r5_d5_lightperf.py` | prepare CLI가 동일 허용집합을 사용하도록 연결. 나머지 승인/실행 로직 보존 |
| 신규 `src/semcache/edgelora/r5_d5_sweep.py` | 12셀 정의, 동결 참조 검증, review-bound CPU 준비, 명시적 subset 제출, 단일 allocation 3조건 순차 실행, append-only 상태 |
| 신규 `src/semcache/edgelora/r5_d5_sweep_analysis.py` | 공식 verifier/compare 결과를 재확인하는 읽기 전용 집계, full-precision CSV/JSON, first10/repeat10 bins |
| 신규 `scripts/163_r5_d5_sweep_v5.py` | Python CLI 진입점. import 시 실행 없음 |
| 신규 `jobs/d5_l2e_bw_ud_sweep_v5.sh` | CPU 제어 명령과 allocated GPU job step을 분리하는 Bash 진입점 |
| `tests/test_r5_d5_lightperf.py` | `.10` 서비스 emulation 및 invalid factor 검사, matrix 수 변경에 대응 |
| `tests/test_r5_d5_lightperf_native.py` | 실제 CPU FULL/EXACT/APPROX의 `.10` logits/rows parity 실행 추가 |
| 신규 `tests/test_r5_d5_sweep.py` | offline UNIT_ONLY 계약·실패·집계 테스트. Slurm/GPU 호출은 mock |
| 본 문서 및 `d5_sweep_v5_diagnostics/` | Phase B 보고와 로컬 로그. 공식 승인/실험 결과가 아님 |

SemCache/모델/물리 재사용/Attention Impact/압축/품질평가/transport 계산식과 공식 `compare`의 identity 필드는 수정하지 않는다. `UDServiceEmulator`도 원문 그대로다. 기존 L2C/L2D/L2E/L3 결과·manifest·receipt를 쓰기 대상으로 사용하지 않는다.

## 2. 0.10 의미와 CPU 검증

허용값은 정확히 `(1.0, 0.5, 0.25, 0.10)`이다. `0.09`, `0.2`, 음수, 0, bool, NaN/Inf 및 문자열을 runtime `knobs`에서 거부한다. CLI float parser/choices도 `.10`을 허용하고 비정상 수치를 거부한다.

실제 LoRA 연산을 한 번 실행해 측정한 비용이 `c`이면 목표 service time은 그대로 `c/factor`다. `.10`은 목표 `10c`, 추가 pacing `9c`이며 하드웨어 CPU throughput/clock 제어가 아니다. Unit clock에서 실제 compute 4 ms, pacing 36 ms, service 40 ms를 확인한다. 결과값은 보존하고 timeout은 예외로 처리한다.

`UD_OPERATION_CALIBRATION_REQUIRED`, 보수적 forecast, `worker_deadline_seconds <= 7200`, Slurm `<=16200`, probability/working/GPU/host/disk bound를 유지한다. `.10`에 대해 부족한 planning bound를 추측하거나 이전 관측 평균을 보수적 상한으로 바꾸지 않는다. `BLOCKED_INFEASIBLE`이면 CPU audit/submit을 진행하지 않는다.

검증 결과:

| 범위 | PASS | FAIL / ERROR / SKIP | 증거 |
|---|---:|---:|---|
| 신규 v5 단위 테스트(전체 회귀 시작 시) | 24 | 0 / 0 / 0 | `d5_sweep_v5_diagnostics/sweep_tests_final.log` |
| 기존 L2C/L2D/L2E/Light/Native 및 v5 관련 전체 회귀 | **269** | **0 / 0 / 0** | `d5_sweep_v5_diagnostics/full_cpu_regression.log`, exit 0, 885.690s |
| 마지막 output-overlap 보호 추가 후 focused 재검증 | **25** | **0 / 0 / 0** | `d5_sweep_v5_diagnostics/sweep_hardening_final.log` |
| Bash syntax | PASS | — | `bash -n jobs/d5_l2e_bw_ud_sweep_v5.sh` |
| 실제 frozen reference | BLOCKED_MISSING_REFERENCE | 미검증 | `d5_sweep_v5_diagnostics/bash_check_local.json` |
| 실제 Slurm/GPU 통합 | NOT_RUN | 미검증 | mock/syntax 결과로 대체하지 않음 |

269개와 focused 25개는 중복이 있으므로 더해 고유 테스트 개수라고 하지 않는다. 전체 회귀가 진행되는 동안 마지막 파일 경로 보호를 보강했고 해당 관리/집계 경로는 focused suite로 다시 검증했다. 수치 연산 코드는 변경하지 않았다. 초기 개발 로그의 fixture 필드 누락 오류도 보존하며 최종 성공 결과와 구분한다. 테스트 실패를 skip이나 PASS로 바꾸지 않았다.

Native suite에는 기존 FULL/EXACT/APPROX 각각 `(50 Mbps, .10)`을 추가해 실제 Torch CPU LoRA/logits/QKV row parity를 확인했다. CPU 테스트에는 localhost TCP만 사용했고 실제 sbatch/scontrol/GPU 호출은 하지 않았다. 전체 repository의 모든 테스트를 실행했다는 뜻은 아니다.

재현 환경은 `CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONPATH=src:tests`이며 기존 `.venv-d5l2e/bin/python -m unittest ... -v`를 사용했다. 선택한 모듈은 `test_r5_d5_sweep`, `test_r5_d5_lightperf`, `test_r5_d5_lightperf_worker`, `test_r5_d5_lightperf_native`, `test_r5_d5_lightperf_replay`, `test_r5_d5_impact`, `test_r5_d5_impact_compute`, `test_r5_d5_owner_diagnostics`, `test_r5_subsequence_validation`, `test_r5_subsequence_generation`, `test_r5_sparse_physical`, `test_r5_natural_approx`, `test_r5_natural_shutdown`, `test_r5_natural_phase3a`, `test_r5_subsequence_audit`, `test_r5_subsequence_diagnostics`다.

## 3. 동결 provenance와 12셀

실제 사용자가 지정한 참조 경로:

```text
SERAPH repo: /data/khuss/repos/GP_semcache_lightperf
PLAN: results/r5/d5_lightperf/l2e_ud025_20261121T083703Z_l2e_plan/plan.json
Audit: results/r5/d5_lightperf/l2e_ud025_20261121T083703Z_l2e_cpu/cpu_validation.json
v4 runner: /data/khuss/tmp/d5_l2e_matched_repeat_v4.sh
```

**로컬 확인 상태: `BLOCKED_MISSING_REFERENCE`.** 위 원본 파일은 로컬에서 읽을 수 없으며 v4 shell 내용도 검증하지 못했다. 실제 seed, asset 경로, contract/null 상태, task hash, model/generation identity 및 operation bound를 임의로 채우지 않았다. 사용자 제공 v4 성공/APPROX 6/20 정보는 역사적 설명이며 로컬 재현 결과가 아니다.

SERAPH에서 `check`는 실제 PLAN inventory/hash와 해당 과거 audit의 source/plan binding을 읽기 전용으로 확인한다. 과거 audit는 해당 과거 PLAN의 증거일 뿐 새 소스의 승인으로 사용하지 않는다. 아래 필드를 실제 원본에서 읽는다:

`original_experiment`, `multi`, `continuous_contract`, `strict_pilot`, `baseline_plan`, `ud_operation_seconds`, `seed`, `stream.tasks_sha256`, `workload_mode`, `enable_continuous_replay`, `impact_implementation` 및 model/tokenizer/semantic/dataset/transport identity.

Null 옵션은 CLI에서 생략한다. Non-null 경로는 실제 absolute path가 존재해야 한다. Inline contract는 새 state에 값 그대로 별도 JSON을 만들 수 있지만, 공식 replay 구현이 non-null contract를 거부하면 그대로 실패한다. 임의로 null로 바꾸거나 replay 계약을 우회하지 않는다. 원본 경로를 재배치/추측/다운로드하지 않는다.

검증된 frozen stream은 20 requests, 처음 10개와 동일한 다음 반복 10개, 8-token cap을 확인한다. `prepare` 후 stream 전체/seed/요청 순서/generation/model/transport를 원본과 비교한다. 차이는 새 source identity, 두 knob 및 그에 따른 resource forecast/plan ID에 한정한다.

state를 만들기 전에 PLAN/audit/original experiment/multi/baseline/pilot/model/dataset 경로와 출력 경로의 ancestor/descendant overlap도 거부한다. original inventory 안에 새 파일을 넣어 과거 데이터를 변형하는 경우를 막는다.

| BW(Mbps) | UD100 | UD050 | UD025 | UD010 |
|---|---|---|---|---|
| 50 | bw050_ud100 | bw050_ud050 | bw050_ud025 | bw050_ud010 |
| 100 | bw100_ud100 | bw100_ud050 | bw100_ud025 | bw100_ud010 |
| 200 | bw200_ud100 | bw200_ud050 | bw200_ud025 | bw200_ud010 |

**12개의 논리적 matrix만 생성했다. 실제 12개 PLAN/CPU audit는 생성하지 않았다.** 일반 Light matrix의 unlimited 포함 16 pairs와 이 v5의 12 cells를 혼동하지 않는다. 기존 official analyze registry 상한(324)은 늘리지 않았으며 셀별 analyze 입력은 3개뿐이다.

## 4. v5 사용법과 실행 계약

다음 명령은 repo root에서 사용한다. `D5_SWEEP_PYTHON`은 활성화된 실제 offline 환경의 Python을 선택한다. 실제 review 값과 cache 경로를 사용자가 제공해야 하며 예시 문자열을 provenance로 등록하지 않는다.

```bash
export D5_SWEEP_PYTHON="$(command -v python3)"
bash jobs/d5_l2e_bw_ud_sweep_v5.sh check --node auto
# 로컬 Slurm manual이 없거나 subset semantics가 확인되지 않으면 auto는 BLOCKED.
bash jobs/d5_l2e_bw_ud_sweep_v5.sh check --node moana-y2
```

새 clean source를 사용자 review/commit한 이후에만:

```bash
: "${D5_REVIEWED_COMMIT:?new user-reviewed clean HEAD}"
: "${D5_REVIEWED_FINGERPRINT:?new approved content fingerprint}"
export D5_SWEEP_STATE="$(pwd)/results/r5/d5_lightperf/sweep_v5_$(date -u +%Y%m%dT%H%M%SZ)"
bash jobs/d5_l2e_bw_ud_sweep_v5.sh prepare-cpu \
  --state "$D5_SWEEP_STATE" --cells all \
  --reviewed-commit "$D5_REVIEWED_COMMIT" \
  --reviewed-fingerprint "$D5_REVIEWED_FINGERPRINT"
bash jobs/d5_l2e_bw_ud_sweep_v5.sh status --state "$D5_SWEEP_STATE"
```

`prepare-cpu --cells bw050_ud025,bw050_ud010`처럼 CPU subset도 가능하다. 셀마다 공식 CLI `prepare → preflight → cpu-validate → inspect`를 실행하고 서로 다른 fresh PLAN/audit 디렉터리를 만든다. 각 audit는 그 PLAN inventory와 knob에 묶는다. 새 소스 source guard는 `--development` 우회 없이 적용한다. 이 과정은 모델/semantic assets가 실제로 있는 offline 환경을 요구하며 12개의 독립 CPU suite를 실행하므로 시간이 오래 걸릴 수 있다.

정상 state 구조:

```text
definition.json                       # immutable reviewed source/reference/matrix
prepared/<cell>/plan/                 # official fresh PLAN + inventory
prepared/<cell>/cpu/                  # that PLAN's fresh CPU audit + inventory
prepared/<cell>/preflight.json
prepared/<cell>/inspect.json
prepared/<cell>/ready.json
attempts/<unique-tag>/submission.json # explicit submit payload, hashed in sbatch args
attempts/<unique-tag>/allocation.json # allocated GPU UUID, CPU affinity, node, order
attempts/<unique-tag>/<condition>/    # three independent native worker sessions
attempts/<unique-tag>/registry.json
attempts/<unique-tag>/analysis/       # official analyze output + inventory
events/<unique-id>.json               # append-only submitted/complete/blocked events
```

기존 state/PLAN/audit/attempt/receipt를 덮어쓰지 않으며 처리 중 실패한 동일 셀을 자동 retry하지 않는다. 원인을 확인한 뒤 필요하면 사용자가 새 state/tag를 명시한다. `status`는 로컬 manifest/event 읽기이며 scheduler 상태를 추측하지 않는다. 실제 pending/running 상태는 job ID를 사용해 Slurm에서 확인한다.

사용자가 실제 submit을 명시 호출할 때만, 첫 canary 순서로:

```bash
: "${D5L_TMPDIR:?existing offline scratch directory under /data/khuss}"
: "${HF_HOME:?existing offline HF cache directory under /data/khuss}"
bash jobs/d5_l2e_bw_ud_sweep_v5.sh submit \
  --state "$D5_SWEEP_STATE" --cells bw050_ud025,bw050_ud010 \
  --trial 0 --node auto --tmpdir "$D5L_TMPDIR" --hf-home "$HF_HOME"
```

이 명령의 동작:

- 선택한 모든 셀의 새 source/PLAN/audit/knob를 **첫 job 제출 전에** 검증한다. 두 canary의 검증된 COMPLETE가 없으면 다른 셀을 거부한다. `.25` canary가 `.10`보다 먼저 와야 한다.
- 정확히 1 cell = 1 job, 1 node/1 GPU/4 CPU/64 GiB/04:30:00, partition `batch_eebme_ugrad`. `--no-requeue`, array 없음. 전체 12셀 일괄 submit 또는 `submit --cells all`은 거부한다.
- Job을 hold 상태로 제출하고 Job ID/payload hash를 먼저 append한 뒤 release한다. 다음 job은 `afterok:<previous-job>`로 연결한다. 큐가 실패/불분명해도 다른 GPU로 재시도하거나 새 job을 몰래 만들지 않는다. release 실패 시 held job은 사용자가 확인해야 한다.
- Node는 `moana-y1,y2,y3,y4,y5,y7`만 허용한다. 직접 지정 또는 auto가 가능하며 y3 고정이 아니다. auto는 설치된 `man sbatch`가 nodelist의 부분 선택 의미를 명시할 때만 6개 허용 목록과 `--nodes=1-1 --ntasks=1`을 사용한다. 확인 불가/구버전 문서는 fail-closed이며 직접 1개 노드를 선택한다.
- SchedMD의 현재 [sbatch 문서](https://slurm.schedmd.com/sbatch.html)는 낮은 node count에서 주어진 nodelist의 subset을 쓸 수 있다고 설명한다. 서버의 실제 Slurm 버전/문서는 로컬에서 확인할 수 없어 별도의 local-man gate와 실제 할당 node 재검사를 둔다. 추측한 feature/constraint 이름이나 다른 node를 추가하지 않는다.
- `srun --cpu-bind=cores`로 1 step을 만들고 그 안의 CPU affinity에서 4개를 고정한다. 세 조건의 Python 자식이 이를 상속하며 실제 receipt의 affinity와 공식 `cpu_concurrency`/backend/GPU/workload equality를 확인한다.
- UUID는 **할당된 job 내부의 별도 PyTorch process**에서 device properties로 읽는다. probe process 종료 후 native 실행을 시작한다. 과거 UUID/visibility를 하드코딩하지 않으며 단일 CUDA, 12 GiB 및 기존 native GPU class/identity guard를 그대로 적용한다.
- 조건 순서는 셀 index에 따라 3가지 순환을 교대하고 `--trial`에 따라 다시 회전한다. 모든 순서를 submission/allocation/event에 기록한다. 각 조건은 공식 `run`의 새 process/owner/session/cache를 사용한다.
- 각 condition 직전에 `scontrol show job --oneliner <job-id>`의 실제 `EndTime`을 읽고 `remaining > worker_deadline+120`, `remaining <=16200`을 확인한다. 부족하면 즉시 `PARTIAL_BLOCKED`다. 세 조건의 최악 7200초가 모두 한 allocation에 들어간다고 가정하지 않는다.
- 각 조건 뒤 공식 `verify`, 3조건 뒤 공식 `analyze` 및 analysis/position/run inventory 재검증을 실행한다. 두 비교 모두 `DESCRIPTIVE_MATCHED_IDENTITY`, EXACT token agreement **20/20**이어야 COMPLETE다. APPROX 6/20 등 불일치는 기록하며 단독 실패 사유로 삼지 않는다. quality 보존은 주장하지 않는다.

Canary 두 셀이 새 소스에서 실제 통과한 뒤 승인된 나머지 subset만 별도 submit한다:

```bash
bash jobs/d5_l2e_bw_ud_sweep_v5.sh submit --state "$D5_SWEEP_STATE" \
  --cells bw100_ud025,bw200_ud025 --trial 0 --node moana-y4 \
  --tmpdir "$D5L_TMPDIR" --hf-home "$HF_HOME"
```

유망 셀의 독립 반복은 사용자가 `--trial 1`, `--trial 2`로 명시 호출한다. 같은 cell/trial 재제출은 거부한다. 최소 3개 독립 allocation 및 균형화된 순서로 확인한 후에만 반복성 판단을 한다. 여러 셀에 동일 GPU UUID가 배정될 것이라고 가정하지 않는다.

## 5. 읽기 전용 집계

```bash
bash jobs/d5_l2e_bw_ud_sweep_v5.sh aggregate --state "$D5_SWEEP_STATE" \
  --output-dir "${D5_SWEEP_STATE}_aggregate_$(date -u +%Y%m%dT%H%M%SZ)"
```

입력 state와 같은 parent 아래의 fresh sibling 출력만 허용한다. 기존 PLAN/audit/run/analysis/position/manifest는 수정하지 않는다. 출력:

- `grid.csv`: 원본 precision의 각 셀·trial·3조건 mean/median/P95 E2E, mean TTFT, FULL/condition session span 및 paired delta, mean E2E delta, measured descriptive speedup, QKV skip/hit, send/recv bytes, metric/quality mean, token agreement, GPU/node/affinity/backend/CPU/source provenance와 comparability.
- `first10_repeat10.csv`: 각 조건의 0:10 / 10:20 구간 E2E/TTFT/skip/hit, TCP send/recv inclusive, owner completion, prefill/decode, INSERT/EVICT. per-request policy delta를 합하며 cumulative counter를 다시 합산하지 않는다.
- `sweep.json`: 위의 full-precision 값과 상세 segment/official comparison/bindings, blocked/unmeasured 셀, exploratory 및 품질 제한을 보존한다. 텍스트/token payload 덤프를 새로 만들지 않는다.

공식 compare가 NOT_COMPARABLE이면 speedup/delta를 확정된 비교 수치로 내보내지 않는다. 임의로 identity 필드를 빼거나 비교 결과를 무시하지 않는다. 잘못된 inventory/position analysis, 누락 run, source/knob 변경은 BLOCKED로 기록한다. 누락 timer는 0으로 채우지 않고 measured/partial/not-measured 상태를 보존한다. Inclusive timer들을 합쳐 E2E를 재구성하지 않는다.

첫 12셀은 단일 시행의 **exploratory descriptive grid**다. 8-token cap과 역사적 낮은 F1은 output correctness 판단을 제한한다. APPROX의 속도/토큰 변화는 품질 보존과 같지 않으며 `correctness_qualified_speedup=NOT_ESTABLISHED`를 모든 결과에서 유지한다. 유망 조건의 독립 3회 이상 확인 이후 Qasper/MultiNews 확장은 별도 승인 범위다.

## 6. Phase B 차단 상태와 보장하지 못하는 항목

1. 실제 frozen PLAN/Audit/v4 runner는 **BLOCKED_MISSING_REFERENCE**. SERAPH에서 원본을 읽기 전 실제 workload/provenance 일치나 `.10` feasibility를 PASS라고 할 수 없다.
2. 현재 worktree는 새 코드로 dirty다. 기존 reviewed commit/fingerprint 및 50/.25 PLAN/audit는 **새 소스의 GPU 승인이 아니다**. 사용자의 새 clean commit/fingerprint 승인 전 `prepare-cpu`가 차단된다.
3. 실제 Slurm client/manual, scheduler submission/hold/release/srun, GPU properties UUID, 실제 node/CPU binding, cache assets 및 EndTime timezone 호환성은 로컬 hardware에서 검증하지 못했다. Unit mock과 Bash syntax PASS는 SERAPH 통합 성공을 보장하지 않는다. 불확실하면 fail-closed다.
4. `.10` forecast가 7200초 제한을 넘거나 세 번째 condition의 남은 allocation 시간이 부족할 수 있다. 상한/timeout/입력/생성 정책을 바꾸지 않고 해당 셀을 중단한다.
5. 강제 종료/노드 장애에서는 Python failure event를 쓰지 못할 수 있다. 기존 official failure receipt, 각 condition budget 기록, Slurm stderr/job ID를 함께 확인한다. 재큐잉/숨겨진 retry는 하지 않는다.

**여기서 Phase B를 종료하고 소스 검토·commit/fingerprint 승인을 기다린다.** Phase C(각 셀 CPU 준비), D(canary submit), E(나머지 subset), F(독립 반복), G(다른 dataset)는 실행하지 않았다.
