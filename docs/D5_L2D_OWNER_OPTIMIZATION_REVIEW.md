# D5-L2D — Cache Owner Completion Profiling & Conservative Optimization

작성일: 2026-11-21, 개인 Ubuntu, CPU-only. **결론: Completion의 비용 경로와 거부 후보의 사전 Proof를 확인했으나, 실패 의미까지 보존하는 최적화를 입증하지 못해 Candidate를 구현하지 않았다.** 기존 Strict/Light 실행 코드는 모두 보존했다. 새 진단 테스트 21개는 PASS지만 Native CPU 검증은 의존성 부재로 BLOCKED이며, 공식 CPU Approval은 생성하지 않았다.

## 1. Baseline 확인

| 항목 | 확인 결과 |
|---|---|
| Branch | `exp/d5-l1-l5-lightperf` |
| HEAD / Reviewed commit | `53407e74f941509fa0b565386b5b0a9b82aaab23` — 일치 |
| Strict baseline | `4916836ce328e119f3a8d85c6c1afc7cf8d363e5` |
| Upstream 대비 | 로컬에 기록된 origin branch와 ahead/behind `0/0`; fetch 미실행 |
| 시작 시 Tracked 변경 | 없음; index와 tracked worktree는 Reviewed commit과 동일 |
| 시작 시 Untracked 변경 | `d5_l2b_inputs.tar.gz`, `results/r5/` — 보존, 출력 경로로 사용하지 않음 |
| 최종 상태 | 새 진단 코드·테스트·보고서·`d5_l2d_diagnostics/`만 추가. 기존 tracked 파일 수정 없음 |
| Python | 기본 `python3`, 3.10.12 |
| 사용 가능 | stdlib `unittest`, ArrayTensor CPU fixture; PyYAML 5.4.1 |
| 사용 불가 | torch, transformers, numpy, pytest, peft |
| 의존성 주의 | PyYAML은 import 가능하지만 프로젝트 선언 `>=6,<7`보다 낮음. 환경 전체가 pyproject 요구를 만족한다고 주장하지 않음 |
| 실행 범위 | CPU fixture 및 localhost TCP 테스트. GPU, Slurm, SERAPH, 원격 접속, 설치, commit, push 미실행 |

`git diff 4916836 HEAD`의 기존 변경은 70개 파일이며 L2B/L2C/Light의 기존 이력이다. 다음 Strict 구성은 baseline과 Reviewed HEAD 사이에도 차이가 없고, 이번에도 수정하지 않았다: `r5_subsequence_physical.py`, `r5_subsequence_integrity.py`, `r5_subsequence_profile.py`, `r5_subsequence_generation.py`, `r5_sparse_qkv.py`, `r5_mstage_bridge.py`, `src/semcache/cache/`.

아래 GPU 수치는 **사용자 제공 D5-L3 결과**다. 로컬에서 재측정하거나 원본 GPU inventory를 재검증한 값이 아니다. 조건은 50 Mbps, UD Throughput 100%, Continuous Replay 20 requests, 기존 Impact Candidate `DEFERRED_HOST_PER_LAYER_V1`, `EXACT_CONTEXT_VERIFIED`, `REQUEST_SCOPED_VERIFIED`다.

| Dataset | FULL / EXACT E2E(s) | Speedup | Impact(s) | Owner(s) | SETUP Tensor Serialization(s) | Materialization nested(s) | Sealing(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2WikiMQA | 101.670 / 105.330 | 0.9652x | 3.027 | 3.657 | 1.337 | 0.418 | 0.212 |
| Qasper | 101.716 / 105.997 | 0.9596x | 2.989 | 4.069 | 1.451 | 0.551 | 0.283 |
| MultiNews | 112.163 / 116.130 | 0.9658x | 2.979 | 2.717 | 1.156 | 0.058 | 0.073 |

단위는 요청당 평균이다. Inclusive/nested 수치를 합산해 Owner나 E2E를 재구성하지 않는다. SETUP Serialization 전체를 Owner 내부 비용으로 취급하지 않는다.

## 2. Cache Owner Completion 호출 구조

아래 라인은 변경하지 않은 Reviewed HEAD 기준이다. 약어:

- P: [r5_subsequence_physical.py](../src/semcache/edgelora/r5_subsequence_physical.py)
- I: [r5_subsequence_integrity.py](../src/semcache/edgelora/r5_subsequence_integrity.py)
- G: [r5_subsequence_generation.py](../src/semcache/edgelora/r5_subsequence_generation.py)
- B: [r5_mstage_bridge.py](../src/semcache/edgelora/r5_mstage_bridge.py)
- C: [global_cache.py](../src/semcache/cache/global_cache.py)
- E: [cache_entry.py](../src/semcache/cache/cache_entry.py)
- M: [metric_manager.py](../src/semcache/cache/metric_manager.py)
- Q: [r5_sparse_qkv.py](../src/semcache/edgelora/r5_sparse_qkv.py)
- F: [r5_subsequence_profile.py](../src/semcache/edgelora/r5_subsequence_profile.py)

정상 완료 요청에 대해 L=layer 수, W=추출된 sliding w=3 windows 수, U=Completion 루프에서 새로 고려하는 unique/nonresident 후보 수, A=정책 승인 후 materialize한 수, S=각 insert 직후 resident여서 seal한 수, V=이 요청에서 실제 전체 검증한 서로 다른 Entry 수로 표기한다. S는 요청 종료 시 resident 수와 다를 수 있다. 앞서 seal한 Entry가 뒤의 삽입에서 퇴출될 수 있기 때문이다.

| 단계 / 함수·라인 | 횟수 / 시점 | Gather·복사 | 전송·Serialization·SHA256 | Admission과 관계 / 안전성 |
|---|---|---|---|---|
| `PhysicalOwner.begin`, P:39–60 | 1/request, 생성 시작 전 | 텐서 Gather 없음 | tensor serialization 없음; request/registry digest는 I:start | bridge 후보 조회 → EXACT position/causal prefix 필터 → 요청 scope 개시. 아직 reuse 허가 아님 |
| `MStageShadow.begin`, B:51–80 | 1/request; lookup 최대 W, metadata 후보 64개에서 중단 가능 | 없음 | metadata 검사, req deepcopy; tensor proof 없음 | registry/physical ownership 확인, overlap selection; `arrive`는 모든 window 출현 빈도와 query clock 갱신. 현재 요청 Admission보다 앞섬 |
| `PhysicalOwner.project/_project`, P:78–140 | prefill 3L회 | fresh Gather, owned scatter, output clone 3L, hidden clone L | output `tensor_proof` 3L회. 실제 CUDA라면 proof가 D2H 유발 | 재사용 여부와 무관하게 full output 유지. projection별 source captures는 완료 후 Admission의 입력 |
| `valid → verify → full → entry_proof`, P:95/116, I:100–120/88–99, P:24–30 | Q eligibility 및 각 역할 소비 전에 valid; scoped full은 V회/request, 각각 3L stored tensors | full 자체 Gather/clone 없음 | 각 tensor serialize+SHA 1회, 이후 metadata/positions/tokens/size aggregate SHA. `fingerprints`는 같은 pass에서 반환 | 과거 Admission seal과 현재 resident object·metadata·storage binding 비교. 요청 시작만으로 full을 하지 않고 최초 사용 시 lazy authorization |
| layer input 검증, P:101–104 | layer당 selected hit별; provenance 부적합이면 생략 | target hidden span Gather 1회 | Gather 결과 serialize+SHA 1회 | 원본 source input proof와 EXACT byte equality. full Entry 무결성 검증과 별개인 수치적 reuse eligibility 경계 |
| `consume`, I:121–138 | 실제 소비마다, accepted hit × role × layer | cache tensor의 private snapshot clone 1회 | snapshot serialize+SHA 1회; metadata/structure 재검사 | full 검증 이후 변조를 검출하고 scatter에 cache mutable alias 대신 인증된 snapshot 전달. scoped 기록은 이미 얻은 actual proof 사용(P:129) |
| sparse protocol, Q:108–129 | Q/K/V마다 full input hash; Q에서 fresh payload 전송 | Q fresh payload Gather; delta deserialize/device copy | protocol serialization/hash, TCP payload는 별도 경로 | Owner proof와 달리 Q/K/V input consistency 및 UD 전송 경계. 본 진단의 `serialized_tensor_bytes`에 protocol bytes는 포함하지 않음 |
| `complete_request`, G:99–104 | 생성 종료 후 1회 | 없음 | TCP finish/ACK metadata hash, 선택적 STOP/BYE | **ACK → 마지막 adapter shutdown → local owner.complete**. ACK/shutdown 실패 시 Owner Admission까지 진행하지 않음 |
| `PhysicalOwner.complete` 진입, P:141–151 | 1회; 모든 projection/생성 완료 후 | 없음 | 없음 | 먼저 validation.close. all-layer capture 및 actual attention 완성 조건 검사. CHU/PBR/Admission과 live authorization이 겹치지 않음 |
| Impact/History/CHU, P:150–161; M:40–43/70–73 | streamed values 조회 1회 또는 fallback reduction W회; history append 1회; consumed된 selected hit별 CHU 1회 | StreamingImpact는 이미 PREFILL에서 계산. fallback attention tensor reduction 가능 | streamed 경로 추가 QKV serialization 없음. fallback GPU reduction은 host scalar를 만들 수 있음 | source 완료 이후, 새 Admission 전. 빈도·Impact 갱신은 실제 소비에 한정 |
| Metadata/Input Proof, P:166–171 | U개 후보 × L회, **승인 전에** | captured hidden span Gather U×L회 | Gather마다 serialize+SHA, metadata에 input_proofs 포함 | 중복 key와 현재 resident key는 건너뜀. 이후 거부될 후보도 input proof 생성. source hidden bytes를 미래 EXACT 판정에 binding |
| `BoundedCache.insert → GlobalCache.insert`, B:28–36; C:73–96; admission.py:20–21 | U회; size/key guard 이후 `admit` 최대 U회 | 승인 전 QKV materialization 없음 | tensor hash 없음; frequency/impact/size normalization과 score | key 존재/개별 size>capacity/strict threshold `score > threshold` 검사. 새 metadata/input_proofs는 정책 score의 피연산자가 아님 |
| materialize callback, P:172–178; E:47–59 | 승인 A회 | captured QKV Gather 3L/entry → `.detach().to(storage_device).clone().contiguous()` 3L/entry | gather·동일 device clone 자체 D2H 없음. device가 다르면 `.to` 전송 가능. 이 경로는 capture device를 지정 | 승인 후, 실제 cache insert 전. `.from_tensors`의 shape·role count·size 계약과 compact owned storage 보장 |
| insert/eviction, C:80–96; B:31–36; P:162–164 | 승인 후 insertion마다, eviction은 필요 수만큼 | storage 회계 조회; QKV Gather 없음 | tensor serialize 없음 | materialized key/size 확인 → insert → byte eviction → 64-entry bound. 스스로 퇴출될 수 있음. EVICT callback이 proofs/admission_bindings 모두 제거 |
| Admission Integrity Seal, P:182–186 → I:88 → P:25 | insert 직후 resident인 S개, 각각 3L tensor proofs | Gather 없음; stored owned tensor 읽기 | stored tensor serialize+SHA 및 aggregate SHA, structure/storage pin 생성 | 승인 후. self-evicted 후보는 seal하지 않음. materialization 이전 tensor proof로 대체할 수 없음 |
| PBR/Completion 종료, P:187–196; M:75–91 | PBR interval일 때 1회, 완료 결과/cleanup 1회 | 텐서 복사 없음; report의 authorization metadata deepcopy는 있음 | PBR은 tensor hash 없음 | PBR은 closed scope에서 policy state 갱신. capacity 확인 후 report/trace, active/pending 해제. abort/dispose도 scope 만료 |

호출 중첩은 `owner_completion_inclusive ⊃ insert_eviction_inclusive ⊃ materialization_nested`이고, sealing은 insert 반환 후 별도로 실행된다. `entry_proof_all_layers ⊃ tensor_proof ⊃ serialization / tensor_sha256`이다. 기존 `owner_completion_ms`는 P:144 이후 P:190까지라서 시작의 `validation.close()` 및 뒤의 `validation.report()`/trace/cleanup을 모두 포함하는 바깥 decorator와 범위가 다르다.

Light의 `generate_native.complete`([timing:151–156](../src/semcache/edgelora/r5_d5_lightperf_timing.py))는 `COMPLETION` location을 별도로 설정하지 않는다. profiler 기본 phase는 `SETUP`(F:20)이다. 반면 Strict generate는 G:175에서 `COMPLETION` location을 설정한다. 따라서 Light의 SETUP 수치를 Strict의 COMPLETION과 그대로 대응시키거나, SETUP serialization을 전부 Owner 내부로 간주하면 안 된다. 출력 proof·protocol·logit transfer도 구분해야 한다.

Validation boundary는 다음과 같이 보존된다.

1. Admission seal: 새로 소유한 저장 데이터와 source metadata의 최초 binding.
2. Request full: 과거 seal 이후의 resident metadata/내용 변조와 object/storage 교체 검출. 요청별로 다시 수행.
3. EXACT target input: 현재 layer input과 source input의 일치 여부. 거부 시 그 span을 fresh로 계산하는 기존 eligibility 규칙.
4. Consumed snapshot: full 검증 이후 실제 사용 bytes의 인증과 alias 분리. 내용/metadata 변조는 hard failure이며 fallback하지 않음.
5. Metadata/structure checks: tensor serialization 없이 header digest, identity/layout/storage, request/thread/epoch, policy state 확인. `_version`만으로 인정하지 않음.

Entry aggregate는 metadata/positions/token IDs/size 및 tensor proofs를 포함한다. frequency/impact/timestamps는 aggregate 밖이며 요청 내 `policy_state`로 고정한다. scope가 닫힌 후 CHU/PBR이 이를 갱신하는 기존 의미도 유지된다.

## 3. 중복 연산 분석

### 관측 방법과 카운터 정의

신규 [CPUOwnerDiagnostics](../src/semcache/edgelora/r5_d5_owner_diagnostics.py)는 기존 Profiler를 상속한다. 명시적으로 `observe()`에 들어간 동안만 process-local 임시 wrapper를 설치하고 `finally`로 복구한다. 기존 파일이나 runtime factory를 수정하지 않는다. 기본 경로에는 import나 timer overhead도 추가하지 않았다. 단일 스레드 CPU 진단 세션만 지원하며 CUDA/HIP visibility를 비우도록 요구한다. CPU tensor guard와 `cuda_events=False`를 사용한다. GPU 진단 모드가 아니다.

기존 tensor serialization과 SHA 결과를 관측하며 추가 tensor hash/복제 없이 아래를 기록한다. 원본 tensor·logits·입력 text/token inventory·object address는 보고서에 덤프하지 않는다. eligibility는 세부 레코드 대신 boundary/reason count만 남기고, cache key는 기존 Profiler의 digest로만 집계한다. Fixture 동등성 비교에는 메모리상의 전체 semantic state를 사용하되 파일에는 aggregate hash만 저장한다.

| 카운터 | 정의 |
|---|---|
| `admission_candidates` | `complete()` 내부에서 실제 `BoundedCache.insert`까지 도달한 후보 수. 기존 resident/동일 key skip은 제외. input proof 이전 실패는 이 수에 포함되지 않음 |
| `admission_approved` | 실제 INSERT event 수. 승인 후 self-eviction도 포함; 최종 resident 수와 다름 |
| `admission_rejected` | INSERT 없이 반환한 수. `CAPACITY`, `KEY_PRESENT`, `POLICY` 구분. 정책을 재실행하지 않고 실제 호출 결과와 기존 guard 조건으로 분류 |
| `candidates[].outcome` | `RESIDENT`, `APPROVED_THEN_EVICTED`, 거부 원인 또는 `EXCEPTION`. RESIDENT는 해당 insert 반환 시점 |
| `tensor_proof_calls` | 기존 Profiler의 tensor_proof scope 횟수. 실패 시도도 포함 |
| `entry_proof_all_layers_calls` | 기존 entry_proof scope 횟수. metadata aggregate와 3L tensor 증명이 포함되는 호출 |
| `request_validation_full_calls` | `RequestValidation.full()` 진입 횟수. `full_entry_full_calls`, `admission_entry_full_calls`를 별도로 유지 |
| 기존 `full_entry/consumed_tensor/admission_entry` | 기존 integrity report의 count/bytes를 변경 없이 유지. consumed는 full과 별개이며 임의 합병하지 않음 |
| `serialized_tensor_bytes` | 성공한 **Physical tensor_proof** blob의 실제 bytes 합. protocol payload, metadata JSON, evidence I/O, logits 제외. 실패 이전/이후 bytes와 numel 예상치를 혼동하지 않음 |
| `serialization_attempts/successes` | CPU guard 통과 뒤 blob 시도/성공 수. profiler tensor_proof 시도와 차이가 날 수 있음 |
| `cpu_serialization_calls` | CPU proof blob 성공 수. GPU D2H 횟수로 해석하지 않음 |
| `gpu_to_cpu_host_transfer_calls` | 본 CPU-only proof 관측에서 0. GPU의 실제 host transfer 수를 추정한 값이 아님 |
| `materialization_calls` | 승인 후 entry materializer callback 호출 수. tensor별 Gather/clone 수가 아님 |
| `completion_gather_calls` | completion 내부 실제 physical Gather 수. 정상 경로 U×L + A×3L |
| `same_content_fingerprint_repeats` | 세션 내 동일 shape/dtype/기존 SHA256 fingerprint의 두 번째 이후 proof 수. 독립 bytes 비교나 불필요성 판정이 아님 |
| `same_live_object_same_fingerprint_repeats` | 동일한 살아 있는 tensor 객체이고 fingerprint도 같은 재증명 수. weak reference로 id 재사용을 구분하며 tensor/storage를 추가로 붙잡지 않음 |

Fingerprint 및 scope inventory는 기존 call budget과 별도 bound로 제한한다. 새 관측값은 [cpu_profile_serial.json](../d5_l2d_diagnostics/cpu_profile_serial.json)의 CPU-only schema로만 저장하며, 공식 Light Receipt의 schema, hash binding, 96 KiB request bound를 변경하지 않았다. 공식 receipt 압축기에 진단 결과를 넣지 않는다.

### 가설 1 — 거부 후보의 Input Proof

**확인됨:** P:170의 U×L input proof는 C:74–77의 capacity/policy 판단보다 앞선다. admission score는 metadata/input proof를 읽지 않지만, 현재 실행의 validation·실패 순서는 그 사전 작업에 의존한다.

정상 Fixture는 L=2, H=4, float32, prompt=6, sliding w=3이므로 후보가 4개다. 각 후보 input proof는 2회, 96 bytes다. 단순한 Admission Reject가 각 layer proof를 생략하게 만들지 않는다.

| Fixture | 후보 / 승인 / 거부 | tensor_proof 전체 | entry_proof / full 호출 | proof 직렬화 bytes 전체 | Completion input proof bytes | Gather(Completion) / materialize |
|---|---:|---:|---:|---:|---:|---:|
| ADMIT | 4 / 4 / 0 | 38 | 4 / 4 | 2,112 | 384 | 32 / 4 |
| CAPACITY_REJECT | 4 / 0 / 4 | 14 | 0 / 0 | 960 | 384 | 8 / 0 |
| POLICY_REJECT | 4 / 0 / 4 | 14 | 0 / 0 | 960 | 384 | 8 / 0 |
| REUSE | 2 / 2 / 0 | 50 | 4 / 4 | 2,880 | 192 | 16 / 2 |
| EVICTION | 4 / 4 / 0 | 38 | 4 / 4 | 2,112 | 384 | 32 / 4 |
| SELF_EVICTION | 4 / 4 / 0 | 20 | 1 / 1 | 1,248 | 384 | 32 / 4 |
| CHU_PBR | 2 / 2 / 0 | 50 | 4 / 4 | 2,880 | 192 | 16 / 2 |
| ALL_PRESENT | 0 / 0 / 0 | 34 | 2 / 2 | 1,920 | 0 | 0 / 0 |

전체 proof 수/bytes는 PREFILL과 Completion을 모두 포함한다. ADMIT는 output proofs 6회/576 bytes + input proofs 8회/384 bytes + Admission stored proofs 24회/1,152 bytes다. 거부 경로의 960 bytes 중 Completion input은 384 bytes이고 나머지 576 bytes는 output proofs다. input과 QKV가 다른 데이터임도 구분해야 한다.

REUSE는 6-token source를 미리 완료한 뒤 8-token target을 실행한다. 초기화/seed는 측정 밖이다. 전체 proof bytes 2,880 중 request full 576, consumed snapshot 576, target input 192, output 768, completion input 192, admission seal 576이다. 동일 숫자가 여러 경계에 나타나도 같은 목적의 증명이라는 뜻은 아니다.

**이동하지 않은 이유:** 새 `test_capacity_rejection_still_raises_on_invalid_input_before_insert`는 capacity=1로 반드시 거부될 후보라도 captured input shape가 손상되면 기존 Gather가 `ValueError`를 발생시키는 것을 관측 OFF/ON 모두에서 확인한다. `complete()` 진입의 scope close는 실행되지만 insert는 실행되지 않으며, 기존 caller의 abort cleanup이 필요하다. Proof를 materialize callback으로 옮기면 이 후보는 조용히 거부되어 실패 의미가 바뀐다. 검증을 앞에 다시 두면 절감이 사라진다. 거부 경로에서 잘못된 tensor를 무시해도 된다는 새 계약을 이번 작업에서 만들지 않았다.

승인/거부별 계측된 input proof 시간은 아래 bin 표와 candidate별 `input_proof.wall_ns`에 있다. 거부 비용을 측정했다고 해서 이 전체 비용을 안전하게 제거 가능하다고 판단하지 않는다.

### 가설 2 — 반복 Serialization

**존재하지만 서로 다른 validation boundary다.** Admission → 다음 request full → 실제 소비 snapshot의 내용을 단순 memoization으로 합치면 storage/metadata 교체 및 full 이후 변조 검출이 약해진다. RequestValidation의 기존 fingerprint export는 이미 entry_proof의 동일 pass를 이용한다(I:88–95, P:28–29). fingerprint를 만들기 위한 별도 tensor serialization pass는 없었다.

| Fixture | `full_entry` | `consumed_tensor` | `admission_entry` | 같은 live 객체/내용 반복 | 같은 fingerprint 반복 |
|---|---:|---:|---:|---:|---:|
| ADMIT | 0 | 0 | 4 | 0 | 35 |
| CAPACITY_REJECT / POLICY_REJECT | 0 | 0 | 0 | 0 | 12 |
| REUSE / CHU_PBR | 2 | 12 | 2 | 0 | 47 |
| EVICTION | 0 | 0 | 4 | 0 | 35 |
| SELF_EVICTION | 0 | 0 | 1 | 0 | 17 |
| ALL_PRESENT | 2 | 12 | 0 | 0 | 31 |

일정한 값을 쓰는 합성 Fixture여서 서로 다른 복사본의 fingerprint가 자주 같다. live-object repeat 0과 content-repeat 35/47을 혼동하지 않는다. 별도 테스트에서는 같은 tensor 두 번, 같은 내용 clone, 원본 변경 후 proof를 실행해 live-object repeat=1, content-repeat=2가 됨을 검증했다. 이 지표는 proof를 재사용하는 승인 근거로 사용하지 않는다.

### 가설 3 — Gather와 소유 복사

**복사는 확인되지만 제거 가능성은 입증되지 않았다.** native `gather`는 device-local `index_select`로 compact tensor를 만들고(Q:15–19), `from_tensors`가 다시 소유 clone을 만든다(E:57). 동일 device의 `.to`는 복사가 없을 수 있고, `.contiguous()`도 이미 contiguous이면 추가 복사를 하지 않는다. 이 호출들을 모두 별도 GPU 전송/복사로 세면 과장이다.

ArrayTensor 경로에서도 Gather 결과와 최종 Entry는 분리되며, source 및 임시 Gather를 수정해도 Entry 내용이 유지되는 테스트를 추가했다. 그러나 ArrayTensor의 Python list 복사 횟수/시간은 Torch storage/allocator/stream 동작의 대체 증거가 아니다. generic `CacheEntry.from_tensors`의 clone을 지우면 호출자가 mutable alias를 유지할 수 있다. PhysicalOwner 전용 ownership transfer를 별도로 설계할 여지는 있지만, shape/role/size 검증과 alias 방지 및 native CPU storage 동등성을 아직 증명하지 못했다. 따라서 해당 경로도 변경하지 않았다.

SELF_EVICTION은 승인 4회, materialize 4회지만 자기 퇴출 3회로 seal은 1회다. 승인 수를 resident/seal 수로 대체하면 이 비용을 놓친다. Eviction을 미리 예측해 materialize를 건너뛰는 것도 현재 예외·storage size 검증·이벤트 순서와 정책을 바꿀 수 있어 시행하지 않았다.

### 세부 Timing과 관측 오버헤드

최종 성능 자료는 **다른 테스트와 겹치지 않게 실행한 30회 반복** [cpu_profile_serial.json](../d5_l2d_diagnostics/cpu_profile_serial.json)이다. 각 시나리오 warmup 후 OFF / 기존 Profiler / 진단 모드 순서를 반복마다 회전했다. seed 작업은 제외했고 입력/설정은 모드 간 동일하다. 초기 20회 자료 `cpu_profile.json`, `cpu_profile_final.json`은 개발 중 탐색 기록이며 아래 표에 혼합하지 않았다.

각 bin은 complete wrapper에서 현재 category를 바꾸어 중첩 시간을 한 곳에만 귀속한다. 기존 Profiler의 inclusive/exclusive 값과는 별개의 partition이며 기존 receipt bin을 수정하지 않는다.

| Bin | 정확한 관측 범위 / 한계 |
|---|---|
| `completion_impact_history` | actual reducer/value retrieval, history.append, metrics.reused. 주변 key/value 조립·trace는 remaining. 이번 Fixture는 기존 테스트의 synthetic impact=1이므로 actual attention math 시간은 검증하지 않음 |
| `completion_input_proof` | materialization 밖 input Gather 및 tensor_proof(직렬화/해싱 포함). metadata 전체 생성 시간은 remaining |
| `completion_admission_decision` | 실제 insert 진입부터 materializer 진입 전까지; 거부 시 insert 전체. 정책 재평가 없음 |
| `completion_materialization` | callback 전체: QKV Gather + from_tensors 소유 복사/검증/metadata 연결. 기존 `physical_materialization` profile은 주로 Gather dict scope로 이보다 좁음 |
| `completion_admission_proof` | `RequestValidation.full(admission=True)` 전체. 그 뒤 structure pins/binding 조립은 remaining |
| `completion_eviction_pbr` | materializer 반환 후 insert/size identity 확인/eviction/entry cap 및 PBR. 순수 eviction만의 시간이 아니라 승인 이후 insert bookkeeping도 포함 |
| `completion_remaining` | 나머지 검사, dictionary 생성/loop glue, CHU trace, binding 구조 생성, report deepcopy, completion trace/cleanup, 해당 구간 observer 비용 |

진단 complete 전체와 7 bins 합의 시작 경계 차이는 `boundary_start_unassigned_ms`로 노출한다. external Fixture completion timer에는 `f.complete`의 synthetic reducer patch 설치/해제도 들어가므로 wrapper 총시간과 같다고 가정하지 않는다. 아래 값은 bin별 **독립 중앙값(ms)**이므로 중앙값들을 합쳐 총 중앙값으로 제시하지 않는다.

| Fixture | impact/history | input proof | decision | materialization | admission proof | eviction/PBR | remaining |
|---|---:|---:|---:|---:|---:|---:|---:|
| ADMIT | 0.0097 | 0.4946 | 0.1482 | 0.4870 | 1.6844 | 0.0425 | 1.0360 |
| CAPACITY_REJECT | 0.0089 | 0.4706 | 0.0192 | 0 | 0 | 0 | 0.3482 |
| POLICY_REJECT | 0.0135 | 0.6509 | 0.1758 | 0 | 0 | 0 | 0.4797 |
| REUSE | 0.0217 | 0.3112 | 0.1127 | 0.2568 | 1.0590 | 0.0273 | 1.5254 |
| EVICTION | 0.0122 | 0.5959 | 0.1584 | 0.5186 | 1.7919 | 0.2127 | 0.9432 |
| SELF_EVICTION | 0.0095 | 0.4758 | 0.1293 | 0.4660 | 0.4218 | 0.1755 | 0.5700 |
| CHU_PBR | 0.0226 | 0.3058 | 0.1214 | 0.2658 | 0.9595 | 0.0790 | 1.4644 |
| ALL_PRESENT | 0.0180 | 0 | 0 | 0 | 0 | 0 | 0.6544 |

External Fixture completion timer의 median [p25,p75], ms:

| Fixture | OFF | 기존 Profiler | Owner diagnostics |
|---|---:|---:|---:|
| ADMIT | 2.2293 [1.9872,2.5296] | 3.2706 [2.9995,3.9040] | 4.2589 [3.9122,4.8354] |
| CAPACITY_REJECT | 0.5411 [0.4914,0.6843] | 0.8057 [0.7117,0.9876] | 1.1263 [1.0162,1.3594] |
| POLICY_REJECT | 0.8934 [0.8053,1.0908] | 1.6060 [1.1937,2.3331] | 1.8026 [1.6501,2.2983] |
| REUSE | 2.2016 [1.9728,2.8345] | 3.1401 [2.4285,3.6940] | 3.7020 [3.0636,4.4106] |
| EVICTION | 2.3452 [1.9810,3.0209] | 3.4926 [2.8583,4.1551] | 4.8569 [4.1305,5.5502] |
| SELF_EVICTION | 1.3612 [1.2602,1.6312] | 1.8099 [1.6383,2.1379] | 2.5769 [2.4243,3.0067] |
| CHU_PBR | 2.3672 [2.0159,2.5883] | 3.1140 [2.6722,3.6450] | 3.6186 [3.1112,4.4393] |
| ALL_PRESENT | 1.1804 [1.0803,1.3559] | 1.2233 [1.0406,1.4159] | 1.0047 [0.9443,1.1861] |

ADMIT의 진단-OFF 중앙값 차이는 약 +2.030 ms지만 **독립 중앙값 차이에 불과**하고 보편적인 observer 보정값이 아니다. ALL_PRESENT에서 진단 중앙값이 더 작은 것 역시 speedup 증거가 아니다. CPU scheduling/clock/cache/trace 실행 분산, 작은 작업, 서로 다른 detail 집계 비용을 포함한다. affinity/CPU frequency/system-wide activity는 통제하지 않았고 신뢰구간이나 GPU 추론을 주장하지 않는다. 전체 진단 context의 설치/해제 overhead까지 포함하는 시간은 JSON의 `whole_observation_ms`에 따로 있다.

## 4. 최적화 Candidate

**구현한 Candidate: 없음 (`candidate=null`). Reference 보존.**

| 평가 대상 | Before / 검토한 After | 절감 가능 연산 | 이번 판정 |
|---|---|---|---|
| 거부 후보 input proof 지연 | 모든 U 후보에서 Gather+proof → 승인 callback에서만 수행 | 이론상 reject×L input Gather/serialize/hash | 거부 전 손상 입력 예외를 없애므로 현재 실패 계약과 다름. 구현하지 않음 |
| proof memoization | Admission/request/consume 각각 검증 → 과거 proof 재사용 | serialization/hash | boundary가 다르고 mutation 검출을 약화함. 구현하지 않음 |
| materialization 소유 복사 | Gather 후 from_tensors clone → 전용 ownership 전달 | 추가 compact clone 여지 | generic 계약의 alias 방지 및 shape/size validation 보존과 native storage 검증이 필요. 미입증, 구현하지 않음 |

따라서 Before/After의 실제 production 동작은 동일하다. 이번에 절감했다고 주장하는 연산/bytes/시간은 **0**이다. 동일 native source에서 안전하게 제거 가능한 중복 serialization을 입증하지 못했다. 향후 이득은 실제 reject 비율, span 크기, L/H/dtype, GPU host transfer, cache occupancy에 의존하며 본 CPU 수치로 GPU 2.717–4.069초 Owner 비용의 절감량을 추정할 수 없다.

tensor_proof, Admission seal, consumed snapshot validation, metadata/storage binding, EXACT 조건, request scope, fail-closed, capacity/admission/eviction, FP64 Impact 순서는 모두 그대로다. 기존 `DEFERRED_HOST_PER_LAYER_V1`도 수정하거나 다른 것으로 대체하지 않았다.

## 5. CPU 테스트 결과

### 결과와 증거

| 범위 | 결과 | 근거 |
|---|---|---|
| 신규 CPU owner diagnostics | **PASS: 21/21**, fail/error/skip 0 | [owner_tests_v2.log](../d5_l2d_diagnostics/owner_tests_v2.log) |
| 최종 관련 회귀 전체 | **210 PASS, 13 FAIL, 0 ERROR, 8 SKIP / 231** | [final_regression_complete.log](../d5_l2d_diagnostics/final_regression_complete.log), process exit=1 |
| Native CPU 동등성/actual attention/real Torch storage | **BLOCKED**, PASS 아님 | torch/transformers/numpy 부재 |
| Reference vs Candidate | **N/A: Candidate 없음** | 비교한 것은 unchanged Reference의 observer OFF/기존 Profiler/진단 ON |
| 기존 조건부 skip | **SKIP 8**, 임의 추가/변환 없음 | 기존 native generation/profile/alias/Linear/reducer/APPROX tests의 dependency decorators |
| GPU / Slurm | **NOT_RUN** | 이번 작업 범위 밖 |
| 공식 CPU Approval | **NOT_CREATED / NOT_APPROVED** | 실패·skip 및 dirty/unreviewed 추가 소스가 있어 승인 조건 불충족 |

13 FAIL은 `test_r5_d5_lightperf_native.NativeParity` 6개와 `test_r5_d5_impact.TorchParity` 7개이며, 기존 setUp이 native dependency 부재를 assertion failure로 처리한 결과다. 처음 Reviewed source로 실행한 기준 재검증에서도 같은 13개가 실패했다([baseline_tests_loopback.log](../d5_l2d_diagnostics/baseline_tests_loopback.log): 168개 중 148 PASS, 13 FAIL, 7 SKIP). 이를 새로운 구현 failure로 오인하거나 통과로 재분류하지 않았다. 최종 범위는 추가 owner 21개 및 공식 목록의 shutdown/phase3a/approx 42개를 포함해 넓어졌다. 전체 repository의 모든 테스트를 실행했다는 뜻은 아니다.

최초 sandbox 테스트는 localhost socket 생성 `PermissionError` 때문에 5 ERROR와 TCP 자식 대기 실패 2개가 추가됐다([baseline_tests.log](../d5_l2d_diagnostics/baseline_tests.log)). CPU localhost TCP 권한으로 재실행한 후 socket 관련 오류는 없어졌다. 해당 실패 로그도 보존했으며 skip 처리하지 않았다.

개발 중 `owner_tests.log`에는 새 harness가 tuple-key history를 JSON digest로 변환하지 못한 오류가 있다. snapshot key normalization을 수정한 후 `owner_tests_fixed.log`, `owner_tests_final.log`, 최종 `owner_tests_v2.log`에서 재검증했다. 초기 오류 로그를 최종 성공 기록으로 취급하지 않는다. 최종 증거와 파일 hash 목록은 [local_verification.json](../d5_l2d_diagnostics/local_verification.json)에 구분한다.

### 동등성 및 실패 경계

- 8종 Fixture, 모드별 30회에서 semantic hash 일치. hash 피연산자에는 timing을 제외한 모든 completion records/output proof, selection에 따른 fresh/reuse rows, integrity report/authorizations, 이벤트(CHU/PBR/INSERT/EVICT), history, cache HIT/MISS, metadata/positions, F/I/timestamps, Entry aggregate proofs, occupancy를 포함한다.
- REUSE/CHU_PBR에서 reused projection rows=36, fresh rows=12, occupancy=1,728 bytes. ALL_PRESENT에서 reused=36, fresh=0, occupancy=1,152 bytes. 여타 정책 결과도 OFF/ON 동일하다.
- 별도 scalar causal generation 테스트에서 OFF/ON 및 기존 EAGER/SCOPED의 Generated Token IDs/Text/전체 logits bytes가 같다. raw logits/text는 파일에 저장하지 않았다. 이는 Tiny OPT/native FP16/FP32 parity를 의미하지 않는다.
- 기존 변조 테스트 11개를 diagnostics context에서도 실행했다: full 이전/이후 QKV 내용 변조, 동일값 tensor/entry 교체와 proof 변경, metadata/positions/size/policy/registry/physical identity/root container 교체, eviction/disposal/abort, 이전 epoch record, 다음 request 재검증, consumed snapshot alias 분리, scope 종료 후 CHU/PBR.
- source/Gather 수정 후 materialized Entry의 데이터가 독립인 것을 ArrayTensor에서 확인했다. 실제 cache eviction 후 proofs와 admission_bindings의 key 집합이 resident entries와 같음을 확인했다.
- 부분 실패에서 observer wrapper와 ContextVar 복구, 원본 예외 전달, reject 전 invalid input 실패 및 caller abort cleanup을 확인했다. 기존 ACK/shutdown 실패·부분 TCP·STOP/BYE·nonfinite generation·independent audit 검증도 관련 회귀에 포함한다.
- native `.data` alias/storage swap 및 실제 attention Impact SHA256/FP64 순서의 검증은 **BLOCKED**다. synthetic history/proof hash 동등성을 실제 Attention Impact SHA256의 native 동등성으로 바꾸어 말하지 않는다.

재현 명령(설치 불필요한 Fixture; 새 output 경로 필요):

```bash
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests \
  python3 -m unittest test_r5_d5_owner_diagnostics -v
CUDA_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/161_r5_d5_owner_diagnostics.py \
  --output /tmp/d5_l2d_owner_cpu_new.json --repeats 30
```

최종 전체 회귀는 위 환경 변수와 함께 다음 모듈을 `python3 -m unittest ... -v`로 실행했다: `test_r5_d5_owner_diagnostics`, `test_r5_subsequence_validation`, `test_r5_subsequence_diagnostics`, `test_r5_subsequence_audit`, `test_r5_subsequence_generation`, `test_r5_sparse_physical`, `test_r5_d5_lightperf`, `test_r5_d5_lightperf_worker`, `test_r5_d5_lightperf_replay`, `test_r5_d5_lightperf_native`, `test_r5_d5_impact`, `test_r5_natural_shutdown`, `test_r5_natural_phase3a`, `test_r5_natural_approx`.

미설치 의존성에 대한 **설치 명령 제안만** 아래에 분리한다. 실행하지 않았고 사용자 승인 전 실행 대상이 아니다. fresh venv에서 프로젝트 선언을 설치하고 실제 resolved version을 새 review/approval에 기록해야 한다.

```bash
python3 -m venv /tmp/d5_l2d_native_cpu_venv
/tmp/d5_l2d_native_cpu_venv/bin/python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu 'torch>=2.2,<3'
/tmp/d5_l2d_native_cpu_venv/bin/python -m pip install -e '.[models,test]' numpy
```

이 명령의 설치 성공이나 native compatibility도 검증하지 않았다. 본 작업에서 패키지를 설치해 문제를 숨기거나 test decorator/setUp을 수정하지 않았다.

## 6. 변경 파일

| 신규 파일/경로 | 내용 |
|---|---|
| `src/semcache/edgelora/r5_d5_owner_diagnostics.py` | CPU-only opt-in temporary observer, 기존 Profiler 재사용, 7개 completion bins, 실제 admission outcome, proof/bytes/weakref identity counters; runtime import 없음 |
| `scripts/161_r5_d5_owner_diagnostics.py` | 기존 tests의 명시적 ArrayTensor fixture 재사용, 8종 시나리오, 순서 회전 반복 및 observer overhead 비교, source hashes, exclusive output creation |
| `tests/test_r5_d5_owner_diagnostics.py` | 21개 진단·동등성·실패·변조 경계 테스트. 기존 안전성 assertion 재사용, 신규 skip 없음 |
| `docs/D5_L2D_OWNER_OPTIMIZATION_REVIEW.md` | 본 보고서 |
| `d5_l2d_diagnostics/` | CPU profile JSON, 초기/최종 검증 로그 및 로컬 증거. 공식 L2C/L3 결과나 Approval 경로와 분리 |

기존 Strict/Light physical/integrity/profile/cache/receipts/timing/runtime/analysis/approval/CLI/jobs 파일 변경은 **없음**이다. 정책·수치 연산·공식 schema/hash binding 변경도 없다. 새 코드 source hashes는 최종 CPU profile에 포함되며 이는 로컬 진단 provenance일 뿐 `CLEAN_REVIEWED_SOURCE`나 CPU Approval을 대신하지 않는다. commit/push는 하지 않았다.

## 7. 후속 GPU 검증 계획

현재 Candidate가 없으므로 Reference/Candidate GPU 비교를 제출할 준비가 된 상태가 아니다. 이번 CPU-only observer는 GPU에 사용할 수 없다. GPU 실행이 필요해지면 별도 사용자 승인 및 아래 절차가 모두 필요하다.

1. **새 Source Review:** reject 실패 의미를 유지하는 설계 또는 ownership 복사 축소 설계를 먼저 입증한다. Candidate는 최대 하나, 명시적인 선택 경로로 만들고 Reference를 보존한다. 새 selector를 PLAN/source inventory/receipt verifier까지 binding해야 하며 기존 impact selector와 구분한다. GPU용 관측을 도입한다면 CPU observer와 별도로 review하고 기존 Light bound/hash schema를 보호한다. 미검토 dirty source나 기존 Reviewed commit을 새 구현 승인처럼 사용하지 않는다.
2. **Native CPU Approval:** 승인받아 dependency 환경을 준비한 뒤 Tiny OPT logits/tokens/text, QKV rows, policies/metadata/proofs/occupancy, 실제 Impact SHA256 및 FP64 순서, `.data`/view/storage mutation, proof binding, scope expiration, partial cleanup, reject/self-eviction을 Reference/Candidate에서 검증한다. 기존 공식 목록과 신규 owner/Candidate 테스트 모두 zero failure/error/skip이어야 한다. 사용자가 나중에 review한 clean source, 새 PLAN 및 inventory에 묶인 fresh approval을 생성한다. 지금의 Fixture JSON과 기존 L2C/L3 approval은 재사용하지 않는다.
3. **2WikiMQA 20-request 비교:** 별도 GPU 실행 승인 이후 동일 RTX 3090/UUID/backend/precision/CPU affinity/seed/입력 stream/50 Mbps/UD100/Continuous Replay 20, EXACT+SCOPED, Impact=`DEFERRED_HOST_PER_LAYER_V1`을 고정한다. Reference/Candidate마다 새 isolated owner/session/cache를 사용하고 warm/cold 조건과 순서를 맞춘다. 원래 W/U/A/S, reject 이유, CHU/PBR/eviction, rows/occupancy/Impact hash/validation 성공을 함께 기록한다. 관측 ON 동등성·오버헤드를 먼저 분리하고 성능 본 비교는 두 경로에 같은 관측 수준을 적용한다. 필요 시 순서를 교차한 반복 trial로 분산을 보고한다.
4. **측정 범위:** E2E/TTFT/prefill/decode와 Owner inclusive, 서로 겹치지 않는 새 세부 bins, proof bytes/calls, 승인/거부별 input cost를 모두 기록한다. GPU stream elapsed, D2H wall, Python overhead를 구분하고 추가 synchronize를 숨기지 않는다. 기존 inclusive/nested timer 합산은 하지 않는다. correctness-qualified speedup은 별도 fidelity 검증 없이는 계속 NOT_ESTABLISHED다.
5. **확장:** 2WikiMQA에서 모든 safety/동등성 조건을 통과한 후 Qasper/MultiNews의 동일 20-request 조건으로 확장한다. FULL_RECOMPUTE도 같은 새 source/환경에서 측정해 EXACT 전용 차이와 E2E speedup을 함께 비교한다.
6. **D5-L3 비교:** 사용자 제공 기존 FULL/EXACT 수치는 역사적 참고로 유지한다. 새 PLAN의 task/stream hash, source/dependency/backend/knobs 및 timer scope가 같은지 확인하고, 우선 같은 새 source의 Reference 대 Candidate paired 결과를 주 비교로 삼는다. 기존 L3와의 차이는 trial/source/observer 차이를 명시한 보조 비교로 제시한다. 기존 L3 결과/approval을 수정하거나 CPU 수치를 GPU 절감량으로 환산하지 않는다.
