# D5 Strict 재실행 호환성 복구

**HASH_COMPATIBILITY_PASS / CPU_FIXTURES_PASS / GPU_NOT_RUN.**

## 작업 범위와 환경

기존 브랜치 `exp/raw-semcache-r5-integration`에서만 작업했다. 로컬 경로는
`/home/songsong/SemCache`, HEAD는 `d38edf85e3d821769e949a0cb1a8f76f2a449aa1`이다.
SERAPH의 현재 커밋으로 전달된 `bca9757f12d99b0f4a12d78b3dca90d6705e604b`와
기준 `83a4fbe26d317b1235e81a7fb22e93676fca5775` 객체는 로컬에 없다. 이를 fetch하거나
브랜치/worktree를 생성·전환하지 않았다. 초기 tracked 상태는 clean이었다.

현재 `communication.py`가 사용자 제공 불일치 SHA와 같음을 먼저 확인했다.
그 후 로컬 과거 Git blob `efad53a34395be02ef0c1f0f462b769c12d9b109`에서
사용자가 지정한 준비 자산 SHA와 정확히 일치하는 바이트를 확인하여 복원했다.
기준83a4fbe 객체를 직접 읽었다고 주장하지 않는다.

## 공통 소스 해시와 D5 전용 예외 처리

| 항목 | SHA-256 |
|---|---|
| 수정 전 | `b2865f648d7a4d505d55470d0ac9beb9b965b5694c9d31a2018de79c0392929c` |
| 준비 자산 요구값 | `a4bf2740338be1d821612f3f0037cf68858721d5923e25368326bf5ccbc02007` |
| 복원 후 | `a4bf2740338be1d821612f3f0037cf68858721d5923e25368326bf5ccbc02007` |

공통 `communication.py`는 준비 당시와 바이트 단위로 일치한다. hash 검사 예외,
manifest whitelist, validation 우회 또는 준비 자산 재생성을 추가하지 않았다.
`r3a_artifacts.validate_prepared()`와 `source_identity()`는 변경하지 않았다.

새 `r5_d5_channel.Channel`은 준비 당시 Channel을 상속한다. 실제 send/recv,
encode/decode, deadline, session/adapter 인증, sequence, tensor shape 및 payload
hash 검증은 원래 메서드에 위임한다. 정상 frame과 성공 Evidence 필드는 동일하다.

실패 시 frozen send/recv의 except에서 `_record()`가 호출될 때 활성 원래 예외를
보존한다. 오류 Evidence를 그대로 기록하며 trace export가 다시 실패해도 원래
예외를 가리지 않는다. legacy handler의 socket close까지 실패하면 원래 예외를
다시 전달한다. failed 상태에서는 재사용이 금지된다. 정상 Evidence export가
실패한 경우도 정상 성공으로 바꾸지 않는다. 가짜 유효 header는 만들지 않는다.

NATURAL runtime의 ES와 CPU UD가 모두 이 D5 Channel을 사용한다. 공통 모듈을
monkeypatch하지 않는다. 따라서 새로운 오류 보존 보장은 D5 NATURAL 경로에
적용되며, 다른 legacy R3-A 호출의 오류 동작을 전역 변경한 것은 아니다.

기존 Adapter 종료 수정은 유지한다:

```text
마지막 Adapter 요청: 실제 completion ACK 검증 -> STOP/BYE 검증/close
                   -> PhysicalOwner.complete/admission -> 결과 저장
조건 종료: 기존60초 join -> UD exit code 확인
```

중간 요청은 기존 completion ACK 후 admission을 유지한다. 마지막 요청에서
STOP/BYE 실패 시 admission을 진행하지 않는다. QKV 계산/재사용, cache 정책,
sharing, generation, numeric thresholds 및 1800초 timeout은 변경하지 않았다.

## 노드 허용과 GPU identity

허용 노드: **moana-y1, moana-y2, moana-y3, moana-y4, moana-y5, moana-y7**.
moana-y6 및 나머지 이름은 거부한다. shell과 runtime 모두 같은 여섯 노드만
허용하며, 실제 `SLURMD_NODENAME`(없으면 단일 `SLURM_JOB_NODELIST`)과
`D5N_NODE`가 정확히 같아야 한다. 다중 노드 표현/불일치/빈 값은 거부한다.

Slurm partition `batch_eebme_ugrad`, nodes1, GPU1, CPU4, mem64G, time04:30:00,
수동 할당·non-array·명시적 GPU opt-in·clean source guard를 유지했다.
`jobs/r5_natural_phase3a.sh`만 확장했고 기존153/157 launcher는 변경하지 않았다.

`gpu_identity.py`와 `r5_i5_gpu.py`는 변경하지 않았다.
`SAME_COMPATIBLE_GPU_CLASS`는 계속 준비/실행 GPU의 RTX3090 class를 검사하고
3090 Ti/A100/unknown UUID/다른 CUDA device를 거부한다. 실행 UUID 기록과
A/B/C 동일 실행 GPU 검증, `cuda:0`, 실제 메모리 guard도 유지한다.
허용 노드 추가는 그 노드의 실제 GPU가 호환된다는 측정/보증이 아니다.

## 변경 파일

* `src/semcache/edgelora/communication.py`: 정확한 기존 바이트로 복원.
* `src/semcache/edgelora/r5_d5_channel.py`: D5 전용 실패 처리 subclass 추가.
* `src/semcache/edgelora/r5_subsequence_natural_runtime.py`: D5 Channel 연결 및
  여섯 노드 검증 helper. 기존 STOP/BYE/owner 완료 순서는 그대로 유지.
* `jobs/r5_natural_phase3a.sh`: 노드 allowlist/안내문만 변경.
* `tests/test_r5_natural_shutdown.py`: 신규 보장을 D5 Channel로 검증하도록 import 변경.
* `tests/test_r5_d5_compatibility.py`: hash/준비 검증/node/GPU guard/transport 호환성 테스트.
* 이 문서. 기존 모델/Adapter, manifest, Run/Plan/CPU approval/Evidence는 수정하지 않았다.

## CPU 테스트

실행 환경은 로컬 `/usr/bin/python3` 3.10.12이다. semcache Conda 환경과
Torch/Transformers/NumPy가 없어 native 테스트는 SKIP이다. 설치나 환경 변경은
하지 않았다. SKIP은 PASS로 계산하지 않는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_d5_compatibility test_r5_natural_shutdown.Logging \
  test_r5_natural_shutdown.Shutdown -v
# 최종 targeted:21 PASS, 0 FAIL, 0 SKIP, 0.315s

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 /tmp/run_d5_compatibility_cpu.py
```

전체 driver는 실제 Python 파일이며 `if __name__=='__main__'` guard가 있다.
Native/multiprocessing.spawn 테스트를 stdin에서 실행하지 않았다. GPU를 숨기고
HF offline을 강제했으며, 실제 localhost socket 권한만 사용했다. job 전체는
실행하지 않았고 node guard 두 조각만 독립 bash 테스트 및 `bash -n`으로 확인했다.

| 그룹 | 실행 | PASS | FAIL | SKIP | 초 |
|---|---:|---:|---:|---:|---:|
| 신규 호환성/노드 | 9 | 9 | 0 | 0 | 0.418 |
| D5 종료/TCP | 13 | 13 | 0 | 0 | 0.150 |
| R3-A protocol/contracts | 16 | 16 | 0 | 0 | 0.735 |
| GPU identity 제어 검증 | 14 | 14 | 0 | 0 | 0.037 |
| Strict/D5/Phase3A | 115 | 106 | 0 | 9 | 8.767 |
| Phase2.5A | 33 | 32 | 0 | 1 | 1.261 |
| 합계 | 200 | 190 | 0 | 10 | 약11.37 |

처음 targeted 실행은21개 중20 PASS/1 ERROR였다(0.792s). 테스트가 reference
JSON에 없는 TEST_FIXTURE_ONLY 필드를 가정한 오류를 수정했다. fixture manifest의
scope와 identity를 검증하도록 바꿨으며, production 검증 코드를 고친 것은 아니다.
최종 로그와 각 SKIP의 ID/사유, 정확한 모듈 목록은 다음 fresh 봉인 결과에 있다:

`results/r5/i5r/d5_natural/phase3a/compatibility_cpu_20261113_01/`

핵심 확인:

* 기대 SHA의 common 파일과 D5가 참조하는 legacy class를 검증했다.
* 실제 수정 없는 `validate_prepared()`가 tiny TEST_FIXTURE_ONLY 준비 자산을
  통과시켰고, 그 manifest bytes를 바꾸지 않았다. 별도 disposable fixture에
  b2865f 해시를 넣으면 기존 hash mismatch 검사에서 거부됨을 확인했다.
* Legacy/D5의 정상 frame bytes·hash·wire counts·Evidence fields가 동일하다.
* header 이전 실패, logger/close 실패, STOP/BYE 실패, failed-completion admission
  방지 및 Runner 오류 전파 테스트를 유지했다. 실제 CPU Sparse TCP 테스트도 통과했다.
* shell/runtime 여섯 노드 허용·y6 거부·실제 할당 불일치·array 거부를 검증했다.
  잘못된 노드는 model setup/출력 생성 전에 거부된다.
* 각 허용 노드에서도 GPU class/UUID/device 검증은 생략되지 않는다(제어면 fixture).
* physical/cache/policy/integrity/auditor/resource/source-pinning 파일의 HEAD 대비
  동일 hash를 receipt에 기록했다.

legacy job shell 실행 테스트 하나는 의도적으로 제외했다. 최종 PASS/SKIP 수에
포함하지 않았다. 나머지10 SKIP은 Torch/Transformers 기반 native 테스트다.
이 결과는 GPU 실행이나 실제 RTX3090 fidelity의 증명이 아니다.

## SERAPH에서 읽기 전용 준비 호환성 확인

실제 기존 MultiNews Plan 또는 expanded preparation이 기록한 `base_prepared`를
`D5_BASE_PREPARED`로 지정한다. 경로를 추측하거나 manifest를 수정하지 않는다.

```bash
cd /data/khuss/repos/GP_semcache
: "${D5_BASE_PREPARED:?actual existing R3-A base preparation directory}"
sha256sum src/semcache/edgelora/communication.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src python -c '
import os
from semcache.edgelora.r3a_artifacts import validate_prepared
manifest,reference,adapter=validate_prepared(os.environ["D5_BASE_PREPARED"])
print({"status":"PREPARED_COMPATIBILITY_PASS", "communication_sha256":
 manifest["code"]["files"]["src/semcache/edgelora/communication.py"]})'

# unittest는 실제 파일/module에서 실행한다. stdin native runner를 사용하지 않는다.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python -m unittest \
  test_r5_d5_compatibility test_r5_natural_shutdown \
  test_r3a_protocol test_r3a_contracts test_r5_i5_gpu -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python -m unittest \
  test_r5_subsequence_generation test_r5_sparse_physical test_r5_subsequence_validation \
  test_r5_subsequence_audit test_r5_subsequence_diagnostics test_r5_natural_approx \
  test_r5_natural_phase3a test_r5_layer_input_verified test_r5_layer_input_audit \
  test_r5_layer_input_native -v
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python -m unittest \
  test_r5_subsequence_natural.Natural.test_native_cpu_three_natural_requests_exact_mode_real_ud -v
```

실제 semcache 환경의 native 결과도 반드시 확인해야 한다. 실제 prepared 자산은
로컬에 없으므로 SERAPH `validate_prepared()` 성공이나 MultiNews Plan02 생성 성공은
**NOT_VERIFIED**다. 단일 common 파일 외의 불일치가 있으면 기존 검사대로 차단한다.
반대로 b2865f 해시로 만들어진 다른 준비 자산은 복원 후 호환되지 않을 수 있다.
그 경우도 manifest를 고치거나 검사를 우회하지 않는다.

## 소스 pinning과 남은 위험

R3-A 준비 fingerprint에는 원래 common 파일이 계속 포함된다. 새 D5 subclass는
그 집합 밖에 있지만, D5 execution fingerprint의 전체 src 집합에 포함된다.
R3-A 입력 호환성 복구와 D5 실행 source 승인은 별도 계약이다.

검토 후 새 D5 소스 commit에 새 파일을 포함해야 한다. 기존 source-pinned D5
Plan/CPU approval은 새 실행 소스에 재사용할 수 없다. 기존 모델/Adapter/원본
sealed workload는 그대로 읽고, 새 source에 pin한 fresh Plan/CPU approval/output을
사용해야 한다. 현재 작업에서는 commit이나 승인을 생성하지 않았다.

새 D5 오류 보존은 frozen legacy error-handler가 `_record()` 안에서 활성 원래
예외를 유지한다는 계약에 의존하며, 그 정확한 legacy 바이트와 호출 경로를 테스트한다.
다른 legacy Channel caller를 전역 변경하지 않는다. 마지막 사용 이후 UD 종료는
유지하지만, 첫 요청 전/중간 요청 사이의 긴 유휴 구간 및 UD I/O 지연과 기존60초
join 한계는 남는다. 실제 GPU 노드 가용성, native CPU, MultiNews GPU 재실행,
Strict numerical audit는 별도 검증이 필요하다.

**GPU/Slurm, 새 branch/worktree, 자동 commit, push, 준비 자산/기존 결과 수정은
실행하지 않았다.** 생성 결과는 새 compatibility CPU 경로뿐이다.
