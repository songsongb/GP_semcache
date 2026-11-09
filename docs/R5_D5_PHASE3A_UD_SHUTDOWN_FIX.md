# D5 Phase3A MultiNews UD 종료 및 오류기록 수정

**CPU_FIXTURES_PASS / RUN217634_CAUSE_NOT_VERIFIED / GPU_NOT_RUN.**

## 작업 환경과 증거 범위

실제 작업 경로는 `/home/songsong/SemCache`, 브랜치는
`exp/raw-semcache-r5-integration`, HEAD는
`efad53a34395be02ef0c1f0f462b769c12d9b109`이다. tracked 변경 없이 시작했으며,
이전 MultiWOZ 변환 작업의 untracked 파일과 기존 logs/results는 유지했다.
새 브랜치/worktree, commit, pull/fetch, push는 만들거나 실행하지 않았다.

지정 SERAPH 루트 `/data/khuss/repos/GP_semcache`, 실패 Run
`results/r5/i5r/d5_natural/phase3a/multi_news_attempts/run_217634`, 원본 실험
Plan은 로컬에 없다. 기준 커밋 `83a4fbe26d317b1235e81a7fb22e93676fca5775`
객체도 없다. 따라서 이 패치는 **로컬 HEAD에 대한 diff**이며, 기준 커밋과의
동일성이나 실제 Run의 요청 순서·마지막 통신 시각을 독립 확인한 결과가 아니다.
서버의 기존 코드·Run·Plan·승인·Evidence에는 접근/수정을 하지 않았다.

## 원인 분석

사용자가 제공한 Run 217634 정보:

* A=FULL_RECOMPUTE COMPLETE, B=EXACT FAILED, C=APPROX NOT_STARTED.
* Adapter_C의 마지막 요청은 P000001T, 32-token 생성 및 완료 HELLO/READY 성공.
* 마지막 UD READY와 ES STOP 사이 1972.123초, UD timeout 1800초.
* Adapter_C UD의 최종 오류는 `KeyError('request_id')`.

이는 timeout 뒤 오류기록이 원래 예외를 가렸다는 가설과 부합하지만, 이 정보는
**사용자 제공 증거**다. 1972.123 > 1800이라는 비교만으로 UD가 어느 구간에서
대기했는지, 실제 원래 예외가 TimeoutError인지 확정할 수 없다. UD log의 원래
traceback, 실제 Plan 및 양 끝 wire ledger 확인이 필요하다. STOP 수신 기록이
없다는 사실도 TCP 바이트가 커널에 도착하지 않았음을 직접 증명하지는 않는다.

로컬 코드 및 CPU 재현으로 확인한 사항:

1. `finish_request()`는 최종 HELLO를 보내고 READY의 completion digest를
   검증한다. 이어 `PhysicalOwner.complete()`가 로컬 Attention Impact,
   CHU, admission/eviction, QKV materialization/증거 저장을 수행한다.
   이 뒤에도 logits/result 저장, scoring, quota scan이 남아 있다.
2. 기존 worker는 **모든 Adapter의 전체 요청**이 끝난 뒤에만 STOP을 보냈다.
   마지막 통신이 끝난 Adapter가 자신의 로컬 완료 작업 또는 다른 Adapter의
   추론·저장을 기다리며 UD recv 상태에 남을 수 있는 구조다.
3. `Channel.recv()`는 prefix/header 수신 전 실패하면 `h={}`로 오류를 기록한다.
   기존 NATURAL ledger의 `record['header']['request_id']`는 이때 KeyError를
   발생시킨다. 기존 except 블록은 trace 실패 시 원래 예외를 재발생시키거나
   channel을 failed/closed로 만드는 코드까지 도달하지 못했다.
4. header 없는 TimeoutError, partial-prefix/header failure 및 실패하는 logger를
   CPU fixture로 재현했다. 수정 후 원래 예외 객체·종류·메시지가 보존되고
   연결은 fail-closed 처리된다. 이는 과거 Run의 원인 확정과 별개다.

## 변경 파일과 종료 순서

* `communication.py`: `_fail()`에서 먼저 failed 상태를 설정한다. 원래 오류의
  종류/메시지를 기록하고, trace/close의 2차 오류가 원래 예외를 대체하지 못하게
  한다. 성공 frame 및 wire serialization, 인증/sequence/shape/hash 검증은 그대로다.
* `r5_subsequence_natural_runtime.py`: 불완전 header는
  `wire/CONTROL/FAILED_FRAME.jsonl`에 격리한다. 이는 **파일 라우팅 이름**일 뿐,
  원래 header에 request_id를 삽입하지 않는다. 정상 request_id는 유지한다.
  Adapter별 계획된 ID 목록의 마지막 요청에서만 STOP callback을 설치한다.
* `r5_subsequence_generation.py`: optional `on_tcp_complete` hook와 작은
  `complete_request()` helper를 추가한다. 기본 None 호출은 기존 완료 순서를
  유지한다. callback은 실제 completion ACK 검증 후, owner.complete 전에 실행한다.
* `tests/test_r5_natural_shutdown.py`: 13개 CPU fixture/TCP 회귀 테스트.

변경 전:

```text
각 요청: 생성 -> 완료 ACK -> PhysicalOwner.complete -> 결과/증거 저장
모든 요청 종료 후: Adapter_A STOP/BYE -> Adapter_B STOP/BYE -> Adapter_C STOP/BYE
                  -> UD 종료 대기/요청 목록/봉인
```

변경 후:

```text
마지막이 아닌 요청: 기존과 동일
해당 Adapter의 마지막 요청:
    생성 -> 완료 ACK -> STOP/BYE 검증 및 socket close
         -> PhysicalOwner.complete -> 결과/증거 저장
조건 끝: 각 UD의 종료를 최대60초 기다림 -> exit code0 확인 -> 요청 목록/봉인
```

실제 ID 순서에서 마지막 사용을 결정하며, 다른 Adapter의 다음 요청을 앞당기거나
뒤섞지 않는다. owner/index/cache는 기존처럼 조건 종료까지 유지한다. admission은
여전히 ACK 뒤 수행하며, 실제 cache 정책/텐서 계산을 생략하지 않는다.
STOP/ACK 또는 UD 종료 실패는 상위 condition 실패로 전파된다. EXACT worker가
실패하면 APPROX를 시작하지 않는 기존 run fail-closed 흐름도 테스트했다.

1800초 frame timeout과 60초 join timeout은 유지한다. retry, heartbeat,
가짜 유효 header, 실패 시 성공 fallback, 근사 허가 우회는 추가하지 않았다.
`r5_sparse_qkv.py`, physical operator, integrity, auditor, model/tokenizer,
LoRA, generation policy/budget, ownership/sharing, resource limits는 변경하지 않았다.

마지막 요청의 STOP/BYE 비용은 이제 request completion/E2E 안에 포함된다.
Profiler에는 `adapter_shutdown_after_completion`으로 나타나며, process join과
최종 봉인은 여전히 request E2E 밖이다. Inclusive scope는 겹치므로 합산하지 않는다.
이전 소스와의 latency를 단순 비교하거나 speedup을 주장해서는 안 된다.

## CPU 테스트 결과

환경: `/usr/bin/python3`, Python 3.10.12, `CUDA_VISIBLE_DEVICES=''`,
`PYTHONPATH=src:tests`. 로컬 Conda `semcache` 환경은 활성화되어 있지 않으며
Torch/Transformers가 없다. 실제 localhost TCP 테스트만 socket 접근을 허용한
실행으로 수행했다. GPU·Slurm·외부 네트워크 작업은 없다.

최종 로그와 봉인된 receipt:
`results/r5/i5r/d5_natural/phase3a/shutdown_cpu_20261110_01/`.

| 최종 그룹 | 실행 | PASS | FAIL | SKIP | 소요 초 |
|---|---:|---:|---:|---:|---:|
| 신규 shutdown/error logging | 13 | 13 | 0 | 0 | 0.066 |
| 기존 R3 TCP protocol | 10 | 10 | 0 | 0 | 0.199 |
| Strict/D5/Phase3A | 115 | 106 | 0 | 9 | 6.868 |
| Phase2.5A | 33 | 32 | 0 | 1 | 1.050 |
| 합계 | 171 | 161 | 0 | 10 | 약8.18 |

신규 테스트는 header 없는/부분 수신 timeout, logger/close 오류의 primary 보존,
정상 request_id, malformed header 거부, send 오류, 마지막 ACK 전 종료 금지,
중복/잘못된 BYE 거부, ACK/STOP 실패 시 admission 금지, UD exit/join 실패 전달,
상위 Runner의 APPROX 미시작, 실제 5-request/3-adapter Sparse TCP를 검증한다.
가상 시계를 1972.123초 진행시키는 테스트는 실제 장시간 대기나 GPU 실험이 아니다.
실제 3-adapter fixture 종료 순서는 C/A/B이며 **실패 MultiNews Plan의 순서가 아니다**.

Strict 그룹 모듈:
`test_r5_subsequence_generation`, `test_r5_sparse_physical`,
`test_r5_subsequence_validation`, `test_r5_subsequence_audit`,
`test_r5_subsequence_diagnostics`, `test_r5_subsequence_natural`,
`test_r5_natural_approx`, `test_r5_natural_phase3a`.
Phase2.5A 그룹은 verified/audit/native 3개 모듈이다.

10 SKIP은 native OPT generation/impact/profile, native Linear/attention,
Torch alias/integrity parity, NATURAL/APPROX native CPU 및 Phase2.5A native
계약 테스트다. 정확한 테스트 ID와 사유는 `cpu_validation.json`에 기록했다.
job shell을 실행하는 기존 `test_shell_requires_manual_allocation_before_execution`
하나는 의도적으로 제외했으며, SKIP/PASS에 포함하지 않았다.

실행 명령:

```bash
# 초기 fixture 점검: 11개 중5 PASS/6 ERROR, 0.023s.
# packet() test helper가 encode의 빈 payload 인자를 빠뜨린 오류였다.
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_natural_shutdown.Logging test_r5_natural_shutdown.Shutdown -v
# helper 수정 후 같은 명령: 11 PASS/0 FAIL/0 SKIP, 0.028s.

CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 -m unittest \
  test_r5_natural_shutdown test_r3a_protocol -v
# 초기에는 imported TestCase도 수집되어29개/27 PASS/2 SKIP, 0.362s.
# module import로 바꿔 중복 수집을 제거했다.

# 상위 Runner 실패 테스트 추가 후 Logging/Shutdown:12 PASS, 0.036s.
# 최종 그룹 실행과 로그/봉인 생성:
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 /tmp/run_d5_shutdown_cpu_tests.py
```

마지막 임시 harness는 최종 그룹 모듈을 unittest로 로드하고 위 job-shell 테스트만
제외했다. 최종 모듈별 선택·소요시간·source hashes는 receipt에 있다. SERAPH에서
신규 테스트는 첫 번째 명령의 두 class 대신 `test_r5_natural_shutdown` 전체를
지정해 실행한다. 이후 기존 native CPU 승인도 실제 semcache 환경에서 수행해야 한다.
CPU fixture PASS는 OPT-2.7B GPU numerical fidelity를 입증하지 않는다.

임시 harness 없이 같은 최종 그룹을 재현하는 CPU-only 명령:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:tests python3 - <<'PY'
import unittest
modules=['test_r5_natural_shutdown','test_r3a_protocol',
 'test_r5_subsequence_generation','test_r5_sparse_physical',
 'test_r5_subsequence_validation','test_r5_subsequence_audit',
 'test_r5_subsequence_diagnostics','test_r5_subsequence_natural',
 'test_r5_natural_approx','test_r5_natural_phase3a',
 'test_r5_layer_input_verified','test_r5_layer_input_audit','test_r5_layer_input_native']
def tests(suite):
    for t in suite:
        if isinstance(t,unittest.TestSuite):yield from tests(t)
        elif not t.id().endswith('.test_shell_requires_manual_allocation_before_execution'):yield t
suite=unittest.TestSuite(tests(unittest.defaultTestLoader.loadTestsFromNames(modules)))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
PY
```

이 직접 명령은 unittest 결과로 SKIP을 출력한다. native 승인에서는 별도로
SKIP=0도 확인해야 하며, SKIP을 통과한 native 검증으로 간주하지 않는다.

## 실패 Run의 실제 순서/시각: SERAPH 읽기 전용 확인

아래 명령은 기존 디렉터리의 JSON/JSONL을 **읽기만** 한다. 실제 사용한 원본
Phase3A Plan 경로를 지정해야 하며 임의 Plan으로 대체하지 않는다. 기존 자료가
없으면 `NOT_AVAILABLE`로 남긴다. 재봉인·backfill하지 않는다.

```bash
cd /data/khuss/repos/GP_semcache
export FAILED_D5_RUN="$PWD/results/r5/i5r/d5_natural/phase3a/multi_news_attempts/run_217634"
: "${FAILED_D5_PLAN:?actual sealed Phase3A plan directory used by Job217634}"
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import json, os
from pathlib import Path
run=Path(os.environ['FAILED_D5_RUN'])
p=json.loads((Path(os.environ['FAILED_D5_PLAN'])/'approx_plan.json').read_text())
last={}
for i,t in enumerate(p['tasks']):
    print('ORDER',i,t['request_id'],t['adapter_id'],t['user_id'],t['sample_id'],t['policy']['max_new_tokens'])
    last[t['adapter_id']]=t['request_id']
cond=run/'SEMCACHE_D5_EXACT'
def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.is_file() else []
for aid,rid in last.items():
    root=cond/aid
    ready=[x for x in rows(root/'ud/wire'/rid/(rid+'_COMPLETE.jsonl'))
           if x['operation']=='send' and x['header'].get('kind')=='READY']
    stop=[x for x in rows(root/'es/wire/CONTROL/shutdown.jsonl')
          if x['operation']=='send' and x['header'].get('kind')=='STOP']
    ud_stop=[x for x in rows(root/'ud/wire/CONTROL/shutdown.jsonl')
             if x['operation']=='recv' and x['header'].get('kind')=='STOP']
    gap=(stop[-1]['monotonic_start_ns']-ready[-1]['monotonic_end_ns'])/1e9 if ready and stop else 'NOT_AVAILABLE'
    print('LAST',aid,rid,'READY_TO_ES_STOP_SECONDS',gap,'UD_STOP_RECORDS',len(ud_stop))
    status=root/'ud/status.json'
    print('UD_STATUS',status.read_text() if status.is_file() else 'NOT_AVAILABLE')
PY
rg -n -C 3 'TimeoutError|timed out|KeyError|request_id' \
  "$FAILED_D5_RUN/SEMCACHE_D5_EXACT/Adapter_C/ud.log"
```

wire record의 monotonic timestamp는 trace 파일 쓰기가 끝난 시각이 아니다.
따라서 위 gap만으로 recv idle/ES admission/NFS 지연을 구분하지 않는다.
원래 exception context 및 각 request timing/profile과 함께 판단해야 한다.

## 남은 위험과 신규 승인

**NOT VERIFIED:** 실제 Run217634의 원래 예외·순서·시각, SERAPH semcache의 native
CPU 계약, GPU numerical fidelity, native throughput/latency, 수정 후 GPU 종료 성공.

이번 수정은 **마지막 사용 이후**의 불필요한 유휴 연결을 제거한다. 첫 요청 전
대기나 같은 Adapter의 아직 남은 요청 사이 공백, UD 자체의 긴 ledger I/O는 여전히
1800초 timeout에 걸릴 수 있다. UD의 봉인이 오래 걸리면 기존60초 join도 실패한다.
이 위험은 자동 timeout 증가/retry로 숨기지 않았다. 실제 GPU 재검증이 필요하다.

소스가 변경되었으므로, 사용자 검토 후 새 commit이 필요하다. 실행 source identity는
commit과 파일 fingerprint 및 clean tracked 상태에 묶인다. 기준83a4fbe로 만든
**기존 D5 실행 Plan과 CPU approval을 수정 소스 실행에 재사용할 수 없다.**
기존 audit/verification 및 실패 Run은 그 소스의 역사적 자료로 그대로 보존한다.
원본 sealed I5 workload/data/adapter 입력은 변하지 않았으므로 읽기 전용으로 재사용하되,
새 commit에 pin한 **fresh D5 Plan -> fresh CPU approval -> 별도 승인 GPU 재실행 ->
fresh analyze/verify**가 필요하다. 새 commit/승인/실험은 이번 작업에서 생성하지 않았다.

receipt는 physical/sparse/integrity/auditor/정책·공유/승인 관련 기존 파일의 HEAD 대비
동일 hash를 기록한다. 원본 실패 디렉터리가 로컬에 없으므로 수정할 수도 없었으며,
생성한 결과는 `shutdown_cpu_20261110_01` fresh CPU 네임스페이스뿐이다.
