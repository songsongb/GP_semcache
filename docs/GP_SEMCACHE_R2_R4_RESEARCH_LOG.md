# GP_semcache 연구기록 — R2 → R3 → R4-A (2026-10-30)

> **목적:** 이 파일 하나로 연구 배경, 단계별 가설·실험·수치·코드 구조·실패 원인·품질 게이트·다음 작업을 복원하기 위한 지속형 컨텍스트.  
> **상태 기준시각:** 2026-10-30 (KST), R4-A **Pilot 실측 완료**, R4-A **Matrix 제출/실행 중이라고 사용자가 보고**. Matrix 결과는 **아직 미수신/미검증**.  
> **대상 프로젝트:** 사용자 본인의 **KV Cache / SemCache / EdgeLoRA 연구**. 다른 공동 사용자와 연관된 IGZO TFT 프로젝트와 **섞지 말 것**.  
> **주 리포지토리:** `songsongb/GP_semcache` (`/data/khuss/repos/GP_semcache`)  
> **주 브랜치:** R3 이력 `exp/raw-semcache-r3` → 현재 R4 구현/실행 `exp/raw-semcache-r4`.  
> **작성 원칙:** `MEASURED`(실제 측정), `ANALYTICAL`(모델 계산), `CONTROLLED_DIAGNOSTIC`(특정 조건 실험), `PLANNED`(계획), `UNCONFIRMED`(원인 미확정)를 엄격히 구분. 성공/FAIL/NOT_RUN을 혼동하지 말 것. 본 파일은 최종 논문 수치가 아니라 연구 진행 스냅샷.

---

## 0. 다른 대화에서 먼저 읽어야 할 핵심 15문장

1. 연구의 큰 질문은 **분산 ES(Edge Server)–UD(User Device) 구조에서 실제 QKV 재사용이 Full Recompute 대비 빨라지는가**, 그리고 품질을 보존하는가이다.
2. 초기 SemCache M1–M9와 CacheGen C1–C9를 이미 진행했다. **현재 R2~R4는 이들과 구분되는 Raw-QKV 중심의 엄격한 물리 검증 계열**이다.
3. R계열의 순서는 **Raw QKV 정확성/오버헤드(R2) → 물리 ES–UD 통신·HIT/MISS(R3) → 실제 자가회귀 생성 E2E(R4)**이다.
4. 우선 모델은 **`facebook/opt-2.7b`, FP16, RTX 3090 24GB, 32 Decoder Layer, hidden=2560**, PEFT/LoRA rank=8, alpha=8, dropout=0, `user_a` synthetic/untrained adapter이다.
5. **R2-A/R2-C:** QKV projection 연산을 크게 줄여도 selector/mapping/snapshot/validation/assembly/Attention/FFN 오버헤드 때문에 Full Recompute보다 느려질 수 있음을 확인했다.
6. **R2-B:** Prefix-KV 경로에서 시간 절약 가능성을 봤지만, **엄격한 FP16 로그잇 동등성 미달** 등으로 정확성 인증된 speedup은 아니다.
7. **R2-D:** M9-A EdgeLoRA 원가/네트워크 모델을 감사한 **CPU-only ANALYTICAL** 결과이며 약 **344 Mbps** 등은 특정 가정하의 이론 수치다. 실제 네트워크의 검증된 임계값이 아니다.
8. **R3-A:** 독립 ES/UD 프로세스가 localhost TCP로 FP16 hidden을 ES→UD 전송하고, UD가 CPU에서 FP32 LoRA ΔQ/ΔK/ΔV를 계산하여 ES로 반환하는 **NO_REUSE** baseline을 만들었다.
9. **R3-A 이후 지속 문제:** distributed CPU-UD LoRA와 Native GPU PEFT가 strict logit parity를 통과하지 못한다. 기준을 임의 완화해서는 안 된다.
10. **R3-B:** GPU-resident **정확한 전체 Prompt(full-prompt) Raw QKV 96개**를 HIT에서 재사용. NO_REUSE와 **bitwise equality PASS**; Native PEFT strict parity는 FAIL.
11. **R3-C/C-F:** userspace loopback TCP proxy로 RTT/BW 영향을 측정했으나 **물리적으로 다른 두 컴퓨터 간 원격 네트워크 실측은 아님**. Natural hit rate도 아님. 조건별 혼합 workload의 break-even은 아직 확정되지 않았다.
12. **R4-A:** R3와 달리 실제 **Prefill→Autoregressive Decode→8/16 token 생성**을 수행한다. **NO_REUSE에서도 `use_cache=True`**로 *동일 요청 내부* 정상 `past_key_values`를 활용한다. 이것이 올바른 baseline이다.
13. **R4-A Pilot 성공:** Job `152025`의 L32_G8 실측에서 Warm HIT vs NO_REUSE **TTFT 578.84→108.36 ms (81.3% 감소), E2E 1800.19→1345.38 ms (25.3% 감소)**; BLEU=100, bitwise cache fidelity PASS.
14. 다만 `NATIVE_PEFT_STRICT_PARITY_PASS=false`, `PRIMARY_PERFORMANCE_ELIGIBLE=false`, `correctness_qualified_speedup=false`이다. **“완전히 정확성 인증된 실서비스 속도 향상”이라고 주장하면 안 됨.**
15. 최신 액션은 `sbatch --export=ALL,R4A_SCOPE=matrix jobs/r4a_opt_full_e2e.sh`로 L32/L128 × G8/G16 **최대 108요청 Matrix** 수행. **결과 아직 받지 않음.**

---

## 1. 목적·범위·역사적 배경

### 1.1 큰 연구 질문

- (Q1) 동일/유사 요청에서 어떤 **QKV**를 *안전하게* 재사용할 수 있는가? `semantic similarity`만으로 Q/K/V 완전 동일성을 보장하지 않는다.
- (Q2) QKV projection 생략 자체가 *어떤 비용*을 줄이는가? Selector, cache lookup, hash, copy, snapshot, mapping, tensor assembly 등으로 **전체 wall-clock이 역전될 수 있다**.
- (Q3) 실제 ES–UD 분리형 EdgeLoRA에서 HIT 시 **통신 왕복과 UD CPU LoRA**를 얼마나 생략하는가? 원격 네트워크 여부, 단말 프로세스와 논리 사용자 여부, 직렬 전송의 가정을 분리.
- (Q4) Prefill만이 아닌 **전체 생성(Decode 포함)**에서도 latency/TTFT 이득이 있는가? Cache 품질과 Native PEFT strict parity를 독립 검증.
- (Q5) 향후 실제 문서/다중 요청에 대해 정답 품질(F1/ROUGE-L/BLEU)과 hit/miss/admission/eviction tradeoff를 확인할 수 있는가?

### 1.2 별도 선행 축 (여기서는 상세 재현 대상 아님)

- **M1–M9 SemCache PoC:** global logical cache, provenance, admission/eviction, EdgeLoRA decomposition, mixed QKV projection skipping, semantic encoder/intent clustering, `w=3` 및 cluster 내부 `exact_token_ids_within_cluster` 제한, 논리적 50-user 및 M9-A ES–UD **분석적 비용 모델**. 실제 50대 사용자 디바이스가 아니며 `safe_reuse_claimed=false`.
- **C1–C9 CacheGen 통합:** Q는 FP16 유지(후기 Q24 실험 별도), K20-bin/V16-bin 등 코덱, 압축 저장/전송, BLEU·token agreement·KL, 실제 GPU codec/latency 및 hit/miss smoke. 압축률로 HIT가 늘어도 품질이 달라질 수 있음. **R3/R4-A에는 CacheGen 없음**.
- 2026-10월 말 연구 우선순위: **Raw QKV 물리적 재사용을 먼저 정확히 검증** → 이후 Cross-model / multi-model KV, 압축/CacheGen, 실제 user/edge 분할로 확대. 서로 다른 축의 수치를 임의로 한 표에 섞지 않는다.

### 1.3 주 실행 환경

| 항목 | 현재 기준 |
|---|---|
| 서버 | 학교 SERAPH, 작업 위치 `/data/khuss/repos/GP_semcache` |
| GPU | NVIDIA GeForce RTX 3090 24GB, Slurm 할당 GPU 1개 |
| Slurm | partition `batch_eebme_ugrad`, `--gres=gpu:1`, CPU 4개, 메모리 64G, 시간 3:00:00 |
| Python | conda env `semcache` |
| Model | `facebook/opt-2.7b`, FP16, 32 layers, hidden=2560 |
| LoRA | Native PEFT `user_a`, rank=8 / alpha=8 / dropout=0 / Q,K,V target. 통제용 **학습되지 않은** adapter |
| HF | 오프라인 캐시 `/data/khuss/huggingface`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| Git | 사용자가 직접 `git push`; **Codex 지시문에 `git push`를 넣지 않는다** |
| 진짜 원격 통신 | **아직 측정 안 함**. R3/R4-A는 독립 프로세스의 실제 loopback TCP 또는 proxy |

### 1.4 부정확한 용어 방지

- **Full Recompute/NO_REUSE (R3/R4):** 요청 **사이** 캐시 HIT를 사용하지 않는 baseline. R4에서는 요청 **내부** Decoding용 일반 Attention K/V Cache가 **항상 활성화**된다(`use_cache=True`).
- **Raw QKV Cache:** Attention의 Q/K/V **projection 직후, head split/scale 전의 원시 출력**. **HF `past_key_values`와 다른 데이터 구조**다. R4에서는 HIT로 raw projection 96개를 재사용하면서 새 요청의 정상 `past_key_values`를 구성·활용해야 한다.
- **Full-prompt HIT:** 모델·LoRA·Token IDs·길이·위치·Context·dtype/provenance가 모두 동일한 완성된 Prompt만 재사용. **Semantic 유사 요청이나 partial-prefix HIT 검증이 아님**.
- **Bitwise fidelity vs Native parity:** `WARM_HIT == distributed NO_REUSE`의 바이트 동등성과 `distributed == native GPU PEFT`의 엄격한 수치 동등성은 **다른 게이트**다.
- **BLEU 100:** 분산 baseline/NATIVE가 실제 생성한 문장을 참조로 비교한 경우 **출력 보존 점수**. 정답(Gold) 없이는 **task accuracy**가 아니다.
- **TTFT:** 첫 생성 토큰이 이용 가능해질 때까지. **E2E wall:** 마지막 토큰 생성, 이후 책무(accounting/admission/finalization)를 포함한 요청 기준. 두 지표를 구분.

---

## 2. 한눈에 보는 단계별 상태표

| 단계 | 목표 | 실제 측정 범위 | 핵심 결과 | 상태/주요 한계 |
|---|---|---|---|---|
| R2-A | GPU QKV 계산 생략의 실제 비용 확인 | OPT FP16, 다양한 length/coverage | QKV skip률 매우 높아도 contiguous/mixed 경로가 느림 | Local compute diagnostic; strict fidelity 문제 |
| R2-B | exact-prefix KV cache + suffix 실행 | OPT, prefix/suffix 길이 | L1024 S64/S1에서 ~1.9x 수준 측정 이력 | FP16 strict FAIL; correctness-qualified 아님 |
| R2-C1~C3 | Raw QKV reuse의 latency breakdown | L512/L1024 | QKV kernel은 감소하지만 impact/validation/copy 오버헤드 큼 | Raw vs Full strict FAIL |
| R2-C4 | 6-arm RAW/Oracle 비용분해 | L512/L1024 | L1024 최대 coverage RAW 869.56ms vs Full 184.60ms | Oracle은 진단용; 실제 구현 아님 |
| R2-D | EdgeLoRA 비용/통신 손익분기 모델 감사 | M9-A + R2-C4 별도 cohort | 344 Mbps 등 **가정적** 임계값 | ANALYTICAL ONLY; cohorts NOT_COMPARABLE |
| R3-A | 실제 ES–UD CPU LoRA TCP NO_REUSE | L32, single forward | socket/adapter split 실행; Native strict parity FAIL | 완전 정합 baseline 아님 |
| R3-A forensic | parity 정밀 진단 | first-layer QKV/FP32/FP64 | CPU precision 3후보 모두 FAIL | 원인 최종 해소 안 됨 |
| R3-B | exact full-prompt physical RAW cache | L32, 21개 요청 | HIT vs NO_REUSE bitwise PASS, warm ~70ms | Native strict FAIL; 생성 X |
| R3-C | loopback proxy RTT/BW | L32, 4 network conditions | 실측 통신·캐시; crossover 미확정 | 실제 remote 아님 |
| R3-C-F | focused break-even | L32, 2 network conditions, 126요청 | C00 12.5% HIT 느림 / 25% unresolved; C03 1-HIT degenerate | Native strict FAIL, primary eligible false |
| R4-A Pilot | 실제 Full Prefill+Decode | L32/G8, 27요청 | TTFT -81.3%, E2E -25.3%, BLEU 100, bitwise PASS | Native strict FAIL, gold/task 품질 없음 |
| **R4-A Matrix** | L32/128×G8/16 길이별 확장 | **최대 108요청** | **아직 결과 없음** | 제출/실행 중, 완료 여부 확인 필요 |

---

## 3. R2 상세 — Raw compute / Prefix-KV / Oracle / Analytical network

### 3.1 R2-A: GPU QKV Computation Skipping과 오버헤드

**연구 질문:** QKV projection을 제외하면 GPU 수행시간도 감소하는가? 실제로는 **projection FLOPs만 줄인다고 요청 wall-clock이 반드시 개선되는 것은 아님**.

- 모델: OPT-2.7B FP16; Full / No-Reuse / Contiguous (부분 갱신/선택) / Oracle 등 조건을 분리한 GPU 실험.
- 보고된 L32 통제 기록: Full 계열 약 **76.09/77.62ms**, Contiguous 약 **140.66/180.83ms** (조건별 수치; 원본 조건 구분 전 합산/평균 금지). QKV skip ratio **98.44%/99.61%**라는 기록이 있으나 Speedup을 의미하지 않음.
- L128 예시 Full **78.94ms**, Oracle **114.17ms**, Contiguous **138.26ms**; L256 Full **79.86ms**, Oracle **127.37ms**, Contiguous **178.69ms**라는 당시 해석 기록.
- 관찰: selector / token mapping / gather/scatter / tensor assembly / contiguous copy / kernel launch / 작은 GEMM / 남아 있는 Attention+FFN 비용 때문에 전체 계산이 증가할 수 있음.
- 수치 출처: 이전 단계 연구 회의·대화 요약. 이 세부 숫자의 원본 실행 JSON은 본 대화에 전부 첨부되지 않아 **논문용 직접 인용 전 R2-A raw artifact 대조 필요**.

**핵심 교훈:** `QKV skip %`는 별도 성능 지표이고, Wall time과 correctness가 최우선.

### 3.2 R2-B: Exact-prefix KV baseline / Suffix reuse

**목적:** 일반 Transformer K/V prefix 재사용으로 이미 계산한 prefix를 피하고, suffix만 처리하여 실측 baseline을 얻기.

- Exact-prefix 재사용에는 **모든 관련 문맥·토큰 순서·포지션·모델 일치**가 중요. 기존 Raw projection cache와 달리 **정상 Attention KV**를 재사용하는 접근.
- R2-B1/B1.5 단계는 실제 Prefix Skip 경로 구현과 엄격 비교를 진행. FP16 full-vs-warm의 **strict parity FAIL**. 예시: L128/P64 max abs **0.208984375**, P127 **0.0234375** (suffix-level 비교 기록); 처음부터 엄격 correctness-qualified 성능 아님.
- R2-B3의 이전 요약: L1024/S64 약 **1.893×**, L1024/S1 약 **1.929×**, 독립 3 Run에서 60/60 timing pair 재현. 후속 B3.5 기록은 L1024/S64 Full 147.58→Warm 81.26ms (**1.816×**), S1 147.04→79.95ms (**1.839×**). **B3와 B3.5 서로 다른 실행 cohort**이며 속도 개선 수치 혼용 금지.
- 엄격 Logit parity 및 일부 출력 shape 이슈가 지속돼 **“correctness-qualified speedup”으로 주장 금지**.

**핵심 교훈:** Full-prompt RAW QKV와 prefix K/V 재사용을 동일하게 취급하지 말 것. R4-B partial reuse에서는 Prefix/포지션 정확성 인증부터 재설계.

### 3.3 R2-C1~C3: Raw-QKV Reuse의 실질 비용

**목적:** 원시 QKV를 재사용하고도 느린 이유를 실제 GPU/host trace로 분해.

- L512/L1024 실험. 높은 coverage에서도 Raw 경로가 Full Recompute 대비 느렸던 기록. R2-C 통제 요약상 **speedup 약 0.279×/0.275×**라는 기록(구체 조건별 원본 확인 후 인용 권장).
- R2-C2 L1024/MAX: QKV kernel **21.52→3.51ms**; 전체 GPU kernel **269.18→235.35ms**로 감소했으나 요청시간 RAW **612.22ms**. Attention Impact 약 **123.7ms**, Snapshot Validation 약 **100ms** 등 고정/검증 오버헤드 관찰.
- R2-C3 Impact ON→REPLAY에서 L512 약 **35–41ms**, L1024 약 **141–170ms** 절감. L1024/MAX에서 RAW 551.78→**382.04ms**로 개선돼도 Full **151.25ms**보다 여전히 **230.79ms 느림**.
- Impact ON vs replay 동일성/검증은 별도 PASS가 있었지만, **RAW vs Full strict FP16 quality FAIL** (특히 coverage>0).

**핵심 교훈:** 실제 시간을 지배하는 overhead를 cache-path component별로 분리해야 한다. REPLAY/Oracle의 비용 절감은 *진단 상한*이지 production 최적화 완료가 아니다.

### 3.4 R2-C4: 6-arm Local Oracle Diagnostic

**목적:** snapshot validation / mapping / 조합 경로의 비용을 Oracle(사후 값 제공)로 제거해 병목 민감도 파악. **Oracle은 구현된 캐시 시스템이 아닌 진단적 상한**.

- 2026-10-29 완료 기록. L512/L1024, RAW baseline 및 여러 Oracle arm. 결과 파일: `cache_path_aggregate_summary.json`, `cache_path_breakdown.json`, `cache_path_decision.json`, `manifest(20261029-112911).json`(첨부 이름); repo에 대응 `manifest.json` 존재.
- L1024/MAX 예시: Full Native **184.60ms**, RAW **869.56ms**, Snapshot Oracle **623.93ms**, Both Oracle **611.03ms**.
- L1024 cov~0.5 (510/1024): RAW−Full **+461.00ms**, RAW−NoReuse **+323.60ms**, Snapshot Oracle 조건부 이득 **121.30ms**, Mapping Oracle **2.34ms**.
- L1024 covMAX (1023/1024): RAW−Full **+684.96ms**, RAW−NoReuse **+536.28ms**, Snapshot Oracle **245.62ms**, Mapping Oracle **8.58ms**, Both Oracle **258.52ms**.
- 관련 L1024 breakdown (nested span, 단순 합산 불가): cov~0.5 `source_snapshot_validate≈118.19ms`, `fast_index_guard≈130.03ms`, `prepared_mapping_get≈0.62ms`, `full_mixed_projection inclusive host≈212.22ms`; covMAX 각각 ≈226.79/127.89/0.92/313.25ms.
- covMAX `contiguous_copy_bytes=502,824,960`은 **GPU측 논리적 copy 카운터**, 실제 네트워크 bytes가 아님.
- **Strict correctness:** coverage>0에서 RAW vs Full **FAIL**, Oracle vs RAW **PASS**. cov=0은 negative control이며 "캐시가 잘 맞는다"는 근거가 아님.

**결론:** Raw reuse의 bottleneck을 계측했지만 현재 local oracle path만으로 Full Recompute보다 빠르다는 주장은 불가능하다.

### 3.5 R2-D: Evidence-Gated EdgeLoRA Distributed Cost & Network Break-Even (Analytical)

- 구현: `scripts/94_run_r2d_edgelora_break_even.py`, `src/semcache/diagnostics/r2d_{evidence,network,analysis}.py`, `r2d_edgelora_break_even.sh` 실행용 스크립트(사용자 업로드로 확인, **현재 GitHub R4 브랜치의 `jobs/r2d_edgelora_break_even.sh` 경로에는 존재 확인되지 않음**). 입력 M9-A summary + R2-C4 aggregate.
- 결과: `r2d_manifest.json`, `r2d_evidence_audit.json`, `r2d_bandwidth_rtt_sweep.csv`, `r2d_break_even_summary.json`, `r2d_decision.json`.
- **실제 실행 성격:** CPU-only 분석; `gpu_inference_performed=false`, `network_executed=false`, `real_network_measured=false`, `full_generation_e2e_measured=false`.
- M9-A 전송 모델: Layer마다 ES→UD hidden (`fresh_tokens×hidden×element_bytes`), UD→ES ΔQ/K/V (`3×fresh_tokens×hidden×element_bytes`). 직렬 payload-only, 일반 메시지 header/실제 RTT/overlap/copy/codec 제외. M9-A 주요 프로파일 L32와 R2-C4 L512/1024는 다른 cohort이므로 **latency를 더하거나 같은 실험으로 비교하지 않는다**.
- 당시 대표 시나리오: L32에서 30/32 재사용, 200 Mbps에서 analytically NO_REUSE 약 **1599.1ms**, REUSE 약 **1022.9ms**; **약 344 Mbps**의 이론적 bandwidth break-even 언급. **실제 전송 측정/검증된 경계 아님**. RTT 가산 시 HYPOTHETICAL, 메시지 round-trip 수 추정치 역시 실측 아님.
- `r2d_decision.json`: `execution_status=COMPLETE`, `classification=ANALYTICAL_DIAGNOSTIC`, `validity_status=VALID_SEPARATE_COHORTS_NOT_COMPARABLE`, `compatibility_status=NOT_COMPARABLE`, `measured_distributed_speedup_status=NOT_ESTIMABLE`.
- **결론:** 실제 ES–UD 프로토콜/왕복·원가·메모리·HIT/MISS·벽시계/생성까지 검증하기 위해 **R3/R4 단계가 필요**하다는 근거.

---

## 4. R3 상세 — 물리적 분산 구조와 Raw QKV cache

### 4.1 R3 전체 원칙/실제 물리 경로

```text
                 Edge Server (GPU RTX 3090)
 Prompt IDs → OPT-2.7B base Transformer + attention/MLP/LM head
                       │
           매 Layer hidden FP16 (MISS / NO_REUSE)
                       ▼
             [ framed localhost TCP ]
                       ▼
                 User Device (CPU 프로세스)
                user_a LoRA A/B 행렬만 보유
             ΔQ, ΔK, ΔV 계산 (통제 FP32)
                       │
           bundled DELTAS FP32 over TCP
                       ▼
                 ES combines base + deltas
                       │
            [R3-B 이후 GPU-resident Raw QKV]
            HIT: prefill QKV 96개 load/reuse
            MISS: 실제 계산 + 통신 + cache admission
```

> 실제 localhost TCP를 사용한 **독립 프로세스**이다. 물리 ES–UD 장치 분리/Internet network 또는 50대 사용자 디바이스를 뜻하지 않는다. R3-A/B/C의 endpoint는 기본적으로 **단일 32-token prefill / one forward-to-logits(`use_cache=False`)**. R4-A부터 autoregressive decode가 추가된다.

### 4.2 R3-A: ES/UD No-Reuse baseline

- 목적: R2-D의 **분산 비용 가정**을 실제 프로세스·socket message·CPU/GPU 연산으로 옮기기. 캐시 없이 `NO_REUSE`만 동작.
- `scripts/95_run_r3a_distributed_no_reuse.py`; `src/semcache/edgelora/{edge_server,user_device,communication,r3a_runtime,collaborative_inference,...}.py`.
- 모델 FP16, L32, batch=1, eager attention, R3 용 `use_cache=False`, Native PEFT separately executed.
- UD는 LoRA adapter weights CPU 보유, ES는 base-only GPU. Layer당 ES→UD HIDDEN 1개, UD→ES QKV DELTAS 1개 **32회 왕복**. 프로토콜 SHA, PID, type/shape/weight fingerprint 등 audit.
- **실측 이슈:** Native GPU PEFT 대 distributed CPU LoRA 정밀도 차이; 기존 대표 숫자 **max_abs=0.0546875**, **relative_l2=0.001902929**, last-token KL≈`1.16e-6`, argmax 일치. 엄격 기준 `max_abs ≤0.01`, `rel_l2 ≤0.001`, `atol=rtol=0.001`, `KL ≤1e-5`, argmax 동의 등에서 **FAIL**. Threshold 임의 완화 금지.
- 오차를 발견했다고 ES–UD 물리 경로가 가짜라는 것은 아니며, **Native 동등성을 주장할 수 없다는 의미**다.

### 4.3 R3-A Precision/Forensics 추적

- R3-A parity forensic/precision experiment를 수행. 주요 첨부: `r3a_cpu_precision_results.json`, `r3a_cpu_precision_summary.txt`, `r3a_cpu_delta_causal.json`, `r3a_cpu_delta_causal_summary.txt`.
- CPU precision 3후보: `CPU_FP32_BASELINE` max_abs **0.0546875**, rel_l2 **0.001902929**; `CPU_FP64_FULL` **0.0625 / 0.001793782**; `CPU_FP64_STAGED` **0.046875 / 0.001854697** → **전부 strict FAIL**.
- 첫 QKV 차이는 Layer 0 Q에서, 다음 layer hidden 차이는 Layer 1부터 전파하는 양상이 측정됨. 이는 첫층 수치 차이의 존재를 보여줄 뿐 **전체 결과의 인과가 완전히 증명된 것 아님**.
- 별도 CPU delta causal/tcp replay 실행은 `execution_status=FAILED`, `INCOMPLETE_FAILED_PROBES`; 원인 최종 `UNRESOLVED`. 이 실패를 정밀도 원인 확정으로 기록하지 않는다.
- 연구 방향: Strict parity를 계속 독립 Gate로 관리하되, 분산 baseline과 그 위의 cache-reuse 간 **동일 환경 bitwise fidelity**를 별도로 검증.

### 4.4 R3-B: 실제 Exact Full-Prompt Raw QKV HIT/MISS

- 기존 R3-A 위에 GPU-resident cache 구축. 핵심 코드 `src/semcache/edgelora/{r3b_cache,r3b_runtime,r3b_analysis}.py`, 실행 `scripts/99_run_r3b_raw_qkv_one_shot.py`, `jobs/r3b_raw_qkv_one_shot.sh`.
- **소유권:** complete request 이후에만 admission. token ids, model, adapter/weight hash, positions, context, dtype/shape/device를 검사. SHA 무결성 확인. **HIT가 아니라면 HIT라고 계산하지 않음**.
- Raw QKV 32 Layer × 3 projection = **96 tensors**, 입력 [1,32,2560], cache owned memory **15,728,640 bytes = 15 MiB**.
- Stage B1 completed cache entry, B2 Cold MISS vs NO_REUSE bitwise, B3 Warm HIT vs NO_REUSE bitwise. 21 요청, audit PASS. CPU UD loopback TCP 실제 실행.
- R3-B one-shot 계측 중 measured 5개/mode 요약: NO_REUSE median **570.65ms**, COLD_MISS **652.74ms**, WARM_HIT **70.37ms**. WARM HIT 데이터 HIDDEN/DELTAS 0 frames, MISS/NR 각각 레이어별 32 frames.
- `r3b_one_shot_results.json`: `output_cache_bitwise_all_pass=true`, Native strict **0 pass / 21 fail**, `PRIMARY_PERFORMANCE_ELIGIBLE=false`.
- **범위:** exact full prompt, single forward; semantic window/partial prefix 재사용 아님, 생성 없음, 실제 remote 네트워크 아님.

### 4.5 R3-C: Network-conditioned loopback TCP

- `scripts/100_run_r3c_network_one_shot.py`, `jobs/r3c_network_one_shot.sh`, `src/semcache/edgelora/{r3c_network_proxy,r3c_calibration,r3c_analysis,r3c_runtime}.py`.
- 실제 socket 송수신은 하되, 사용자 공간 proxy로 bandwidth/RTT-like impairment 주입. **원격 두 머신 통신 아님**, 특권 `tc` 네트워크 장비 실험도 아님.
- 대표 조건: `C00=(RTT 0ms, unlimited)`, `C01=(0ms,200Mbps)`, `C02=(10ms, unlimited)`, `C03=(10ms,200Mbps)` (상하향 bandwidth 동일).
- R3-C 1회실험 228 request rows, wire SHA/independent process audit PASS, native strict FAIL; condition별 HIT ratio crossover `UNRESOLVED`.
- 성능 수치 단순 비교 시 conditioning, latency endpoint, request schedule과 cache fill 포함 여부를 정렬해야 하며 `R2-D` 예측 숫자와 합산 금지.

### 4.6 R3-C-F: focused break-even follow-up (2026-10-30)

- `scripts/101_run_r3c_break_even_focused.py` (GitHub `exp/raw-semcache-r4`에서 존재 확인), `jobs/r3c_focused_break_even.sh`.
- 실제 Job **151913**의 총 **126요청**: Primary 96, auxiliary measured 16, primary warmup 6, auxiliary fills 2, 기타 검사/controls. 네트워크 조건 C00 및 C03 집중.
- Primary 4 arm, 각 cache 16요청 + 짝지은 16 NoReuse 측정. 아래 `Δ = cache mean − NO_REUSE mean`, **양수=cache 느림**.

| Condition | HIT 비율 | NO_REUSE mean | Cache mean | Δ(ms) | 95% CI (ms) | 해석 |
|---|---:|---:|---:|---:|---|---|
| C00 0ms/unlimited | 2/16=12.5% | 571.17 | 613.74 | +42.58 | [+10.33,+73.74] | CACHE_SLOWER |
| C00 | 4/16=25% | 571.17 | 573.65 | +2.48 | [-31.39,+41.38] | UNRESOLVED |
| C03 10ms/200Mbps | 0/16=0% | 2238.42 | 2362.02 | +123.60 | [+112.84,+135.47] | CACHE_SLOWER |
| C03 | 1/16=6.25% | 2238.42 | 2211.43 | -26.99 | [-35.19,-18.65] | **1-HIT degenerate; 통계적 강한 판정 금지** |

- Auxiliary **resident-only** HIT n=8: C00 mean **74.10ms**, C03 mean **71.15ms**. Conditional cache HIT는 뚜렷하게 빠름. 하지만 혼합 요청에서 fill/miss/reset 포함 손익분기는 별개.
- 평균 혼합 모델의 탐색적 break-even 추정: C00 약 **22.24%**, C03 약 **5.24%**. **실제 측정된 명확한 bracket/root 아님**, C03 1-HIT 문제 존재. R2-D 이론 344 Mbps와 직접 비교 금지.
- 결과 `r3cf_arm_statistics.json`, `r3cf_break_even.csv`, `r3cf_break_even_decision.json`, `r3cf_results.json`, `r3cf_requests.jsonl` 등.
- **최종:** `PRIMARY_PERFORMANCE_ELIGIBLE=false`, Native strict FAIL, 혼합 workload robust threshold 미확정.

---

## 5. R4-A 상세 — OPT Full E2E Generation + Raw QKV Reuse

### 5.1 R4-A 설계 및 중요한 공정성 규칙

- 모델/LoRA/프로토콜: R3 exact-full-prompt Raw QKV 구조를 그대로 활용, **CacheGen 압축 비활성**.
- 세 Mode: `NO_REUSE`(cross-request bypass), `COLD_MISS`(새 계산+완료 후 cache insert), `WARM_HIT`(동일 complete prompt의 GPU resident QKV reuse).
- 세 Mode **모두 `use_cache=True`**, 정상 Attention `past_key_values`로 **동일 요청 내 토큰별 incremental decoding**. R4-A의 NO_REUSE는 모든 과거 토큰을 매 Decode마다 재계산하는 불공정 baseline이 아님.
- 첫 생성 토큰은 prefill logits, 나머지 G−1 토큰은 각각 1-token decode; 각 decode step은 UD LoRA 계산/TCP 왕복 유지. HIT가 재사용하는 것은 **Prefill 96개 Raw QKV**만이며 attention/MLP/LM head/Decode는 계속 계산.
- Fixed greedy generation (`do_sample=False`, max_new_tokens 8 또는 16, EOS 무시 fixed-latency 모드); Native와 distributed는 동일 policy.
- 전송/정확성 검증: prefill 0 HIDDEN/DELTA on HIT, decode 정상 프레임, hash/position/cache length/shape/dtype/device/LoRA provenance, NaN/Inf fail-closed.
- 핵심 경로 `scripts/102_run_r4a_opt_full_e2e.py`; `src/semcache/edgelora/r4a_{generation,dynamic_ud,session,cache,quality,analysis,evidence,runtime,fixture,parity,diagnostics}.py`; `docs/R4A_OPT_FULL_E2E.md`; `jobs/r4a_opt_full_e2e.sh`.
- **현재 브랜치는 `exp/raw-semcache-r4`**; 이전 Codex 초안에는 `r3` hardcoded preflight가 있었으나 `gpu_preflight()`가 `r4`인지 변경·확인 후 실행.

### 5.2 2026-10-30 초기 실패 Pilot 151991와 원인 진단

- 최초 Slurm Job **151991**, `results/r4/a/opt_full_e2e_151991_20261030T114408Z/`.
- `r4_native_prepare` 중 `ProtocolError: nonfinite actual model logits`; traceback 위치 `r4a_generation.py` `ForwardDriver.last_logits`.
- 당시 Prompt는 실제 단어 토큰 수가 15–17인데 좌측 PAD로 `L32`를 채움: text_0 **17+15**, text_1 **17+15**, text_2 **16+16**, text_3 **15+17** (real+PAD).
- `triage_native_0`(use_cache=False), `triage_native_1`(use_cache=True)는 완료됐으나 이후 Native 실제 생성 때 실패. **NaN이 padding 위치 때문에 생겼다는 것은 미확정**: 당시 코드는 모든 위치 logits를 검사했고 실패 위치/step 추적이 없었음.
- Codex 수정: 서로 다른 충분히 긴 deterministic source에서 정확한 **32/128 실제 token IDs**, PAD 0, attention mask all ones를 강제. `r4a_suite_v2`와 로딩 시 재검증. NaN/+Inf/−Inf 진단을 all/valid/padded/final 위치별 저장. HF OPT positional lookup + 실제 KV 길이/shape/dtype/device 강화. 수치 FAIL을 clamp/무시하지 않음.
- 수정 후 CPU 테스트 31 PASS/4 SKIP, 관련 R3 90 PASS/5 SKIP (legacy-fixture 초기 단발 오류 후 별도 재현 PASS). 기존 831 source/evidence 파일 보존 보고.
- **결과:** 다음 새 Pilot 성공으로 이전 failure를 막는 입력/경로 개선은 실증됐지만 과거 PAD-NaN의 수치적 인과는 여전히 확정 아님.

### 5.3 2026-10-30 성공 Pilot 152025: 정확성/성능 기준선

**Job:** `152025` (GPU host `moana-y4`, RTX 3090).  
**경로:** `results/r4/a/opt_full_e2e_152025_20261030T123056Z/`  
**실험 Cell:** `L32_G8`, 실제 입력 token=32, 출력 token=8.  
**스케줄:** 4개 Prompt × 2 repeat blocks × 3 modes = 24 measured + 3 warmups = **27 requests**; 27×8 = **216 generation steps**. Batch 1, loopback ES–UD socket, sequential request walls.  
**모델:** Native PEFT 및 distributed base/LoRA strict parity 별도 측정, synthetic **untrained adapter**. Gold reference 없음.

**각 Mode의 평균 실측** (각 measured n=8):

| Metric | NO_REUSE | COLD_MISS | WARM_HIT |
|---|---:|---:|---:|
| TTFT mean (ms) | **578.8404** | 572.4726 | **108.3627** |
| E2E wall mean (ms) | **1800.1865** | 1870.8499 | **1345.3789** |
| Sequential generated tokens/s | 4.44398 | 4.27613 | 5.94628 |
| Sequential requests/s | 0.55550 | 0.53452 | 0.74329 |

- **Warm vs NR:** TTFT **−470.48ms / −81.3% / 5.34×**, E2E **−454.81ms / −25.3% / 1.34×**. **두 Repeat Block 내 총 8개의 matched pair 모두 유리**.
- **Cold vs NR:** E2E **+70.66ms** (느림). 부가 insert 비용·lookup 검증을 간과하면 안 됨.
- **One Cold + One resident Warm**의 요청당 평균 `(1870.85+1345.38)/2≈1608.11ms`, NO_REUSE 1800.19ms 대비 **약 10.7% 낮음**. 단, 독립적으로 측정한 2×NoReuse batch를 대조한 것은 아님.
- 간단 혼합 평균 공식 가정 `p_hit ≈ (T_cold−T_nr)/(T_cold−T_hit)` → 약 **13.4%** 탐색 임계값. **실제 자연 발생 HIT 분포의 인증된 break-even 아님**, R3-C-F의 조건별 22.24%/5.24%와 서로 다른 측정 계열.
- 요청별 애플리케이션 전송량 (사용자에게 분석 보고된 요약): NO_REUSE **45.29MB**, WARM_HIT **8.52MB**, 약 **36.77MB / 81.2% 절감**. Prefill HIT 때 **0 HIDDEN / 0 DELTAS frames**, 나머지 7개 Decode Forward는 정상 통신. **원격 전송량이라 부르면 안 됨**.
- 생성 `cache`/`NoReuse` 16 비교(`WARM` 8, `COLD` 8)는 **생성 token IDs/문자열·logits SHA·96 QKV SHA 등 bitwise equal PASS**, ES/UD wire SHA agree. Native도 generated output token/text match **100%**, 하지만 strict logit parity 실패.
- 출력 보존 BLEU: `bleu_vs_distributed_full_recompute=100.0`, `bleu_vs_native_peft=100.0` (8 samples/mode). **SacreBLEU 2.6.0**, corpus BLEU, `tok:13a`, `case:mixed`, `smooth:exp`, `eff:no`, nrefs=1, range 0–100. **이 값은 fidelity; task quality 아님**.
- `bleu_vs_ground_truth.status=NOT_ESTIMABLE_NO_GOLD_REFERENCES`.
- NaN/Inf: 개선된 32-token non-padded workload의 216개 generation step에서 실패 없이 완료, run audit `requests=27`, `generation_steps=216`, `real_socket_forward_passes=209`(HIT에서 제외된 prefill + 별도 triage 포함), independent wire SHA PASS.
- GPU memory result peak allocated 약 **5.42GB** (Pilot scope에서의 계측; 128 또는 긴 context extrapolation 금지).

**Gate table (R4-A Pilot)**:

```yaml
execution_status: COMPLETE
R4A_PHYSICAL_E2E_PASS: true
R4A_RAW_QKV_E2E_FIDELITY_PASS: true
R4A_TTFT_BENEFIT_OBSERVED: true
R4A_E2E_BENEFIT_OBSERVED: true
NATIVE_PEFT_STRICT_PARITY_PASS: false
R4A_GOLD_BLEU_AVAILABLE: false
PRIMARY_PERFORMANCE_ELIGIBLE: false
correctness_qualified_speedup: false
actual_remote_network_measured: false
concurrent_serving_validated: false
trained_adapter: false
cachegen_enabled: false
```

**Native strict FAIL 상세:** 8 measured NO_REUSE reference 요청의 8 steps=**64 step comparisons 모두 strict FAIL**. Argmax는 일치하고 generated tokens도 일치하지만 `max_abs` 최대 **0.046875**, `relative_l2` 최대 약 **0.0041484**, 허용 기준 `max_abs≤0.01`, `rel_l2≤0.001`. 이것은 기존 R3-A의 수치 정합 실패가 **전체 생성에서도 지속**된다는 증거. 임계값을 낮추지 말 것.

**통계 주의:** `r4a_analysis.py::blocked_contrast()`은 **2개 repeat-block mean**으로 10,000회 block bootstrap. 보고된 Warm HIT E2E CI `[-457.23,-452.38]ms`, TTFT `[-475.22,-465.73]ms`는 **독립 배치가 2개뿐**이라 지나치게 확신하면 안 된다. 8/8 paired improvements는 직접적인 사실이나, 독립 Slurm job/GPU node·여러 Prompt 반복이 필요하다.

**논문에서 안전한 표현:** “OPT-2.7B/RTX3090/32-token input/8-token decode, exact-full-prompt cache resident reuse에 대해, 동일 distributed NO_REUSE 대비 generated output의 bitwise fidelity를 유지하면서 TTFT/E2E 감소를 관찰했다. Native PEFT strict parity 미충족으로 correctness-qualified production speedup claim은 보류한다.”

### 5.4 R4-A Matrix — **현재 실행 중 / 결과 미수신**

사용자에게 확인된 제출 명령:

```bash
cd /data/khuss/repos/GP_semcache
git switch exp/raw-semcache-r4
git pull --ff-only
mkdir -p logs
bash -n jobs/r4a_opt_full_e2e.sh
sbatch --export=ALL,R4A_SCOPE=matrix jobs/r4a_opt_full_e2e.sh
```

- 총 실험 Cell: `L32_G8`, `L32_G16`, `L128_G8`, `L128_G16`.
- 각 Cell: 4 Prompt × 2 blocks × 3 modes + 3 warmups = 27 requests; 총 **최대 108 requests**, 최대 1296 forward generation steps.
- `R4A_SCOPE=matrix` 명시 필요 (GitHub 당시 launcher default `matrix`였지만 환경을 명확히 표시). `squeue -u khuss` / `logs/r4a_opt_e2e_<JOB_ID>.out`에서 실제 완료/중단 확인.
- 결과 받을 때 **새 Job ID/UTC 경로를 반드시 적고**, 이전 Pilot의 결과와 동일 파일명끼리 혼동하지 말 것.
- 범위 제한: `L128`은 실제 128토큰이며 padding 0; 4K/8K/LongBench 아직 아님. Network impairment 조건/원격 환경을 Matrix에서 자동 수행한다고 가정하지 않음. Actual codec/semantic sharing/멀티유저도 아님.
- **현재 기준으로 Matrix의 PASS/FAIL·성능 수치를 지어내지 말 것.**

---

## 6. 데이터·평가·논문 연결: 앞으로의 로드맵 (PLANNED)

### 6.1 R4 다음 가설 및 권장 순서

1. **R4-A Matrix 분석:** 입력 L32/128와 생성 G8/16의 TTFT/E2E 비교 및 최소 repeated-job 재현; Native strict parity 그대로 표기.
2. **Native 정합성 해결 R4-A0/R3-A follow-up:** first differing QKV layer, FP32 CPU vs FP16 GPU PEFT rounding, LoRA delta addition order, position/cache interaction; tolerance 고정. 원인 미확정 상태에서 조작하지 말 것.
3. **R4-B exact-prefix/partial reuse:** position/context에 의존하는 QKV를 다른 Prompt에서 그대로 재사용하면 안전하지 않을 수 있음. **exact full-prompt의 성공을 partial/semantic safety로 외삽 금지**. Prefix-KV/R2-B의 parity 실패 고려.
4. **R4-C LongBench:** OPT 파일럿 후 더 긴 Context를 지원하는 **Qwen3-4B** 검토. QA `Qasper`, `2WikiMQA`, `MuSiQue`; 요약 `MultiNews`, `SAMSum`. 공식 QA F1/EM, ROUGE-L 등 + Corpus BLEU/출력 보존 병기. Gold references 필요.
5. **R4-D SCBench:** 동일 Context의 multi-request, cache retention/admission/eviction/natural hit-rate와 accurate serving. 후보 `scbench_kv`, `scbench_qa_eng`, `scbench_prefix_suffix`. **아직 실행하지 않음**.
6. **R4-E 모델 범용화:** Qwen3-4B, 자원 허용 시 Mistral-7B 등을 CacheBlend/SGLang 지표에 맞춰 검토. 모델마다 QKV shape, GQA, LoRA target, rotary/absolute positional handling이 달라질 수 있어 R3/R4-A 코드를 무검증 포팅 금지.
7. 별도 이후 축: **Edge/UD 실제 물리 분리**, bandwidth/RTT, Multi-user concurrency, raw vs compressed QKV(CacheGen), cache budget, hit ratio, energy·memory tradeoff.

### 6.2 참고 논문과 맡길 역할

| 논문/벤치 | 프로젝트에서 참고할 부분 | 그대로 재현 여부 |
|---|---|---|
| SemCache | Semantic-aware KV sharing, EdgeLoRA/ES–UD 구조, 정책·품질 개념 | core mechanism PoC 우선. 원 논문 모든 수치 재현 아님 |
| SGLang | Exact prefix reuse, Serving hit/latency/throughput baseline | 나중에 통제된 Serving 비교 |
| Prompt Cache | 모듈 prompt reuse 및 위치/문맥 이슈 | 개념·경계 조건 참고 |
| CacheBlend | Partial/selective KV recomputation; 2WikiMQA, MuSiQue, SAMSum, MultiNews | 정답품질 F1/ROUGE-L, TTFT 함께 비교 |
| LongBench | 긴 Context를 가진 실제 QA/요약 data + 원 데이터셋 공식 metrics | R4-C 계획 |
| SCBench | Shared Context/multi-turn/multi-request 캐시 유지와 지연/품질 | R4-D 계획 |
| CacheGen | 압축 저장·전송 및 Rate–Quality tradeoff | R4-A 제외, 후속 확장 계획 |
| E2LLM | Global/Query/Segment 캐시 파티셔닝과 네트워크 비용 | 실제 UD/ES 분할 설계 아이디어 |

### 6.3 BLEU와 품질 판정의 세 구분

- **Output-fidelity BLEU:** Hypothesis=cache HIT 생성 텍스트, Reference=distributed full recompute 생성 텍스트. 100이어도 두 결과가 같은 것만 증명; 모델이 질문을 잘 답했는지는 미확인.
- **Native-fidelity BLEU:** Hypothesis=distributed, Reference=Native PEFT. 토큰/텍스트 일치가 logits strict parity PASS를 뜻하지 않음.
- **Gold BLEU + 공식 데이터셋 Metric:** 실제 정답을 reference로 사용. QA라면 F1/EM, 요약이라면 ROUGE-L 등 *원 데이터셋 공식 평가*가 핵심; BLEU는 보조. Data leakage / 여러 정답 / text normalization 고려.
- SacreBLEU `signature` 및 `sample_id` 정렬, nrefs/tokenizer/문자 정리/생성 정책/문장 수를 항상 저장. 이전 C6/C8 BLEU 숫자와 R4-A BLEU 100은 **평가 목적이 다르므로 수치 비교 금지**.

---

## 7. 재현·실행·분석 파일 지도

### 7.1 주요 코드/문서와 실행 명령

```text
GP_semcache/
├── scripts/
│   ├── 94_run_r2d_edgelora_break_even.py       # R2-D CPU-only analysis
│   ├── 95_run_r3a_distributed_no_reuse.py      # R3-A prepared Native baseline
│   ├── 98_explore_r3a_cpu_precision.py         # R3-A parity diagnostics/process
│   ├── 99_run_r3b_raw_qkv_one_shot.py          # R3-B physical raw cache
│   ├── 100_run_r3c_network_one_shot.py         # R3-C socket proxy
│   ├── 101_run_r3c_break_even_focused.py       # R3-C-F, GitHub 확인됨
│   └── 102_run_r4a_opt_full_e2e.py             # R4-A runner
├── src/semcache/edgelora/
│   ├── edge_server.py, user_device.py, communication.py
│   ├── r3a_runtime.py, r3b_{runtime,cache,analysis}.py
│   ├── r3c_{runtime,analysis,calibration,network_proxy}.py
│   ├── parity_forensics_probe.py
│   └── r4a_{generation,dynamic_ud,session,cache,quality,analysis,evidence,runtime,fixture,parity,diagnostics}.py
├── jobs/
│   ├── [R2-D launcher: 사용자 업로드 파일, repo 경로 확인 불가]
│   ├── r3a_edgelora_no_reuse.sh
│   ├── r3b_raw_qkv_one_shot.sh
│   ├── r3c_network_one_shot.sh
│   ├── r3c_focused_break_even.sh
│   └── r4a_opt_full_e2e.sh
└── docs/
    ├── R2D_EDGE_LORA_NETWORK_BREAK_EVEN.md
    └── R4A_OPT_FULL_E2E.md
```

> 위 코드 지도는 기능 기억용이다. `scripts/101_run_r3c_break_even_focused.py`는 `exp/raw-semcache-r4`에서 확인됨. R2-D launcher `r2d_edgelora_break_even.sh`는 이 대화에 첨부된 사용자 실행 파일이고 **R4 브랜치의 `jobs/` 경로는 미확인**. 실제 Checkout에서 `git ls-files | grep -E 'r2d|r3c|r4a'`로 다시 확인.

### 7.2 R4-A 결과 아티팩트

```text
results/r4/a/opt_full_e2e_<SLURM_JOB_ID>_<UTC>/
  base_prepare/                   # R3-A model/base/native input preparation
  prepare/native/                # Native PEFT references + native logit diagnostics
  run/                            # top-level run exports
    r4a_results.json
    r4a_decision.json
    r4a_quality.json
    r4a_requests.jsonl
    r4a_tokens.jsonl
    r4a_latency_contrasts.csv
    r4a_sacrebleu_signatures.json
    r4a_parity_diagnostics.json
    es/                           # actual ES/UD wire & triage files
      r4a_logit_diagnostics.jsonl
  aggregate/                      # independent post-analysis
```

- `r4a_decision.json`: Gate 및 cell별 latency summary, native strict failures.
- `r4a_latency_contrasts.csv`: paired Δ, CI, n, block 수. 표기된 CI가 지나치게 좁아도 **두 블록 부트스트랩**이라는 한계 명시.
- `r4a_quality.json`: BLEU vs distributed/native 및 gold reference 유무.
- `r4a_results.json`: 실제 GPU/프로세스/파일 해시와 transport audit.
- `r4a_requests.jsonl`: 요청별 mode/sample/block/TTFT/E2E/cache admission/통신 bytes.
- `r4a_tokens.jsonl`: 각 생성 step의 KV 길이, QKV SHA, frames, logits, position 및 진단 정보.
- 실패 시 `prepare/native/stderr.log`, `native_result.json`, `r4a_logit_diagnostics.jsonl`, `launcher_failure.txt` 확인.

### 7.3 기존 첨부에서 직접 참고 가능한 결과 파일

- R2: `r2d_decision.json`, `r2d_break_even_summary.json`, `r2d_bandwidth_rtt_sweep.csv`, `r2d_evidence_audit.json`, `r2d_manifest.json`, `cache_path_aggregate_summary.json`, `cache_path_breakdown.json`, `cache_path_decision.json`.
- R3-A: `r3a_cpu_precision_results.json`, `r3a_cpu_precision_summary.txt`, `r3a_cpu_delta_causal.json`, `r3a_cpu_delta_causal_summary.txt`.
- R3-B: `r3b_one_shot_results.json`, `r3b_one_shot_summary.txt`, `r3b_requests.jsonl`.
- R3-C: `r3c_one_shot_results.json`, `r3c_network_calibration.json`, `r3c_requests.jsonl`, `r3cf_results.json`, `r3cf_break_even_decision.json`, `r3cf_arm_statistics.json`, `r3cf_break_even.csv`.
- R4-A Pilot: `r4a_decision.json`, `r4a_quality.json`, `r4a_latency_contrasts.csv`, `r4a_requests.jsonl`, `r4a_tokens.jsonl`, `r4a_results.json`.
- 설계 프롬프트 기록: `R2D_Codex_Prompt.txt`, `R3A_Codex_Prompt.txt`, `R3B_OneShot_Codex_Implementation_Prompt.txt`, `R3C_Network_OneShot_Codex_Prompt.txt`, `R3C_Focused_BreakEven_Codex_Prompt.txt`, `R4A_OPT_Full_E2E_Codex_Prompt.txt`.

**출처 식별 주의:** 여기 나열한 단일 파일 이름은 **이 대화에 업로드된 artifact basename**이다. SERAPH에서는 `results/rN/...` 아래 여러 Job에서 같은 basename이 반복될 수 있으므로 **항상 Job ID와 UTC 디렉터리로 cohort 고정**.

---

## 8. 미해결 문제·중요 결론·다음 대화용 체크리스트

### 8.1 지금 주장 가능한 것

- Raw QKV의 단순 projection skipping과 전체 시스템 speedup은 다르며, R2의 오버헤드 계측에서 이를 확인했다.
- R3-B는 독립 ES/UD 프로세스와 localhost TCP 사이에서 **exact full-prompt GPU-resident Raw QKV physical reuse**가 동작하며, distributed NO_REUSE 대비 **bitwise output fidelity**를 유지했다.
- R3-C/C-F에서 network proxy를 포함한 조건별 hit/miss의 경제성은 단순하지 않았다. Cold fill, 실제 HIT ratio, round-trip/전송과 실행 스케줄을 모두 포함해야 한다.
- R4-A 성공 Pilot은 **실제 autoregressive generation을 포함한 TTFT/E2E 감소**와 동일 distributed baseline 대비 출력의 bitwise equality를 **좁은 통제 조건에서** 보였다.

### 8.2 아직 주장하면 안 되는 것

- Native GPU PEFT와 분산 CPU LoRA 구현이 strict numeric parity를 통과한다.
- SemCache 논문의 숫자를 재현했거나 모든 semantic hit가 안전하다.
- Prefix/substring/유사 문서 QKV를 아무 문맥에서나 안전하게 재사용할 수 있다.
- 실제 독립 Edge/UD 단말 간 물리 네트워크에서 동일 효과가 난다.
- CacheGen 압축을 R3/R4-A에 이미 적용했다.
- BLEU 100이 실제 QA/요약 정답 품질을 보장한다.
- R2-D의 344Mbps 또는 R3-C의 추정 HIT threshold가 현실 배치의 확정 기준이다.
- R4-A Matrix 108요청 결과가 PASS했다 (아직 수신/확인 전).
- 2-block bootstrap의 매우 좁은 CI만으로 일반적 통계적 유의성을 확정했다.

### 8.3 다음 대화에서 실제로 할 일

**최우선: R4-A Matrix 결과 확인**. 사용자로부터 새 Job의 `r4a_decision.json`, `r4a_latency_contrasts.csv`, `r4a_quality.json`, `r4a_results.json`, `r4a_requests.jsonl`, `r4a_tokens.jsonl`을 받는다.

확인 순서:
1. Job ID, UTC 경로, `execution_status`, `completed_cells`가 4개 모두인지. 조기 종료/partial result 여부.
2. `TEST_FIXTURE_ONLY=false`, 실제 GPU/ES+UD socket 사용 및 독립 SHA 감사, Native reference 정상 실행.
3. R4-A `R4A_PHYSICAL_E2E_PASS`와 `R4A_RAW_QKV_E2E_FIDELITY_PASS` 둘 다 true인지. **Native strict parity는 별도 표기**.
4. L32_G8 vs L32_G16 vs L128_G8 vs L128_G16, mode별 **TTFT, E2E, Decode-related overhead, wire bytes, cold insert**, 비율/차이 및 8개 paired deltas.
5. 세 Mode의 fixed policy, 동일 prompt 및 all-one mask, in-request `past_key_values` 길이 정상 여부.
6. `quality` BLEU vs distributed, vs Native, Gold 미제공임을 검증.
7. 여러 cell을 합산해 “전체 100% HIT 속도 향상” 같은 표현을 하지 말고 **조건별 결과** 발표.
8. GPU peak memory, longer 128-token source에 대한 자원/성능 skew, 3시간 Slurm 제한 확인.
9. Native parity strict FAIL을 먼저 분석할지, 독립 반복/LongBench·SCBench 데이터셋 후속을 먼저 할지 결정.

**Codex 작업 습관:** 기존 R3 및 과거 evidence 해시 보존, 수치 임계값 사후 완화 금지, 실패도 evidence 보존, CPU fixture PASS≠실제 GPU 성공, **`git push`는 사용자가 직접 수행**.

---

## 9. 짧은 연구 서사 (향후 서론/발표에 재사용)

> SemCache 방식의 캐시 재사용은 계산 생략률만으로 성능이 정해지지 않는다. 첫 번째 R2 실험에서는 QKV 계산을 상당 부분 생략해도 인덱싱·스냅샷 검증·텐서 조립과 잔여 Attention/FFN 비용 때문에 Full Recompute보다 느릴 수 있다는 것을 확인하였다. 이후 R2-D에서는 분석적 ES–UD 전송 모델의 가정과 실측의 한계를 분리하고, R3에서 이를 실제 CPU LoRA User Device 및 GPU Base Edge Server의 TCP 메시지 경로로 구현하였다. R3-B의 완전히 동일한 Prompt에 대한 resident Raw QKV 재사용은 분산 No-Reuse와 bitwise 동일한 결과를 유지하면서 Prefill 통신을 생략하였고, R3-C에서는 RTT·대역폭·Cold Fill·HIT 비율의 영향을 확인하였다. R4-A에서는 이를 실제 autoregressive generation으로 확장하여, 32-token 입력·8-token 생성 Pilot에서 Warm HIT의 TTFT 81.3% 및 E2E 25.3% 감소를 관찰하였다. 다만 분산 CPU LoRA와 Native GPU PEFT 간 엄격한 numerical parity 문제가 남아 있으므로, 해당 성능 관찰은 통제된 분산 baseline 대비 진단 결과로만 해석하며, 향후 longer context, real datasets, natural multi-request workloads에서의 correctness-qualified 비교가 필요하다.

---

## 10. 버전 기록

- **2026-10-30 / v1.0:** R2-A/B/C/C4/D, R3-A/B/C/C-F, R4-A failed Pilot 151991 + successful Pilot 152025, R4-A Matrix pending을 현재 대화의 업로드 자료, 이전 연구 문맥, 리포지토리 코드/문서 및 실측 artifact를 대조하여 정리.
- **TODO when Matrix arrives:** v1.1에서 Job/시간/네 Cell의 TTFT/E2E/quality/Native parity/네트워크 bytes/통계 한계를 업데이트하고 `R4-A Matrix` 상태를 명시적으로 `COMPLETE`, `PARTIAL`, `FAILED` 중 하나로 전환. 현재 값으로 결과를 예단하지 말 것.
