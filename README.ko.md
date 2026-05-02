# MoE-MADV: 64GB M1 Max에서 284B MoE 모델 돌리기

[English README](README.md)

이 저장소는 **DeepSeek V4 Flash**라는 284B 파라미터 MoE 모델을
64GB 메모리의 Apple Silicon 컴퓨터에서 실행해 본 실험 기록입니다.

핵심은 이겁니다.

- 모델 전체 크기는 Hugging Face 기준 약 **150GB**입니다.
- 테스트한 컴퓨터 메모리는 **64GB**입니다.
- 모델 전체를 메모리에 다 올릴 수 없기 때문에, 필요한 부분을 NVMe/파일 캐시에서 가져오며 실행합니다.
- 이때 MoE expert가 필요한 순간에 맞춰 OS에 `MADV_WILLNEED` 힌트를 주면 디코드 속도가 좋아졌습니다.

![Decode speed headline](docs/assets/deepseek-q4-decode-headline.svg)

## 이 프로젝트를 한 줄로 말하면

큰 MoE 모델을 작은 메모리 머신에서 돌릴 때, 병목은 계산만이 아니라
**필요한 expert 가중치를 얼마나 빨리 불러오느냐**가 됩니다.

`MoE-MADV`는 Mixture-of-Experts 모델에서 선택된 expert의 파일 페이지를
미리 준비하도록 `MADV_WILLNEED`를 적용해 본 실험입니다.

## 핵심 결과

가장 중요한 결과는 디코드, 즉 실제 답변을 생성하는 속도입니다.

| 모드 | 사전 로드 | expert 페이지 힌트 | 디코드 생성 속도 | 걸린 시간 |
| --- | ---: | --- | ---: | ---: |
| 기본 실행 | 끔 | 끔 | 0.98 tok/s | 115.5초 |
| 최적화 실행 | 끔 | `MADV_WILLNEED` | 1.23 tok/s | 103.1초 |

정리하면, 모델을 바꾸지 않고도 디코드 생성 처리량이 **약 25.4% 증가**했고,
디코드 전체 시간은 **약 10.7% 감소**했습니다.

## 초보자를 위한 배경 설명

일반적인 LLM은 대부분의 가중치를 반복해서 사용합니다. 한 번 메모리에 올라온
가중치가 계속 재사용되기 쉬운 편입니다.

하지만 MoE, 즉 Mixture-of-Experts 모델은 다릅니다. 토큰마다 여러 expert 중
일부만 선택해서 사용합니다. DeepSeek V4 Flash는 이런 expert 라우팅 때문에
생성 중에 필요한 가중치 위치가 계속 바뀔 수 있습니다.

그래서 150GB 모델을 64GB 메모리에서 돌리면 이런 문제가 생깁니다.

- 모델 파일은 디스크/NVMe에 있습니다.
- 실행 중 필요한 부분만 메모리로 들어옵니다.
- 다음 토큰에서 다른 expert가 선택되면 또 다른 부분을 불러와야 합니다.
- 이 파일 페이지 로드가 느리면 GPU/CPU 계산보다 I/O 대기가 더 큰 병목이 됩니다.

이 프로젝트는 그 병목을 줄이기 위해, MoE 라우팅 결과로 선택된 expert 범위를
계산 직전에 OS에 알려주는 방식을 실험했습니다.

## 어떤 모델을 사용했나

최종 벤치마크에 사용한 모델은 다음 파일입니다.

- Hugging Face 저장소:
  [`lovedheart/DeepSeek-V4-Flash-GGUF`](https://huggingface.co/lovedheart/DeepSeek-V4-Flash-GGUF)
- 파일:
  `DeepSeek-V4-Flash-MXFP4_MOE.gguf`
- 정확한 파일 URL:
  `https://huggingface.co/lovedheart/DeepSeek-V4-Flash-GGUF/blob/cd42deba41ac0536e68b125dfc367197b0ec3038/DeepSeek-V4-Flash-MXFP4_MOE.gguf`
- 기반 모델:
  [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- 로컬 파일 크기:
  `150,225,324,672` bytes, 약 `139.91 GiB`
- Hugging Face 표시 크기:
  `150 GB`

모델 파일은 이 GitHub 저장소에 포함하지 않았습니다. 너무 크기 때문입니다.
대신 다운로드 스크립트와 정확한 모델 주소를 문서에 적어 두었습니다.

자세한 모델 출처와 시도했던 다른 모델 목록은
[docs/model-sources-and-parsers.md](docs/model-sources-and-parsers.md)에 있습니다.

## 먼저 필요한 것

이 저장소를 그대로 재현하려면 꽤 큰 로컬 디스크 공간이 필요합니다.

- macOS / Apple Silicon 환경
- 여유 디스크 공간 최소 200GB 이상 권장
- 최종 GGUF 모델만 받을 경우 약 150GB 필요
- MLX/safetensors 원본까지 받아 `packed_experts_q4`를 만들 경우 추가로 150GB 이상 필요
- Python, CMake, C/C++ 빌드 도구
- Hugging Face 모델 다운로드가 가능한 네트워크

64GB 메모리에서 돌아가긴 했지만, 빠른 모델은 아닙니다. 이 프로젝트의 목적은
"작은 메모리에서 대형 MoE 모델을 빠르게 서비스한다"라기보다,
"메모리보다 큰 MoE 모델을 어떻게 로컬에서 실행하고 병목을 줄일 수 있는지"를
확인하는 데 가깝습니다.

## 빠르게 실행해 보기

저장소 루트에서 실행합니다.

```bash
# patched llama.cpp 런타임 준비
scripts/setup_deepseek_gguf_runtime.sh
```

모델을 다운로드합니다.

```bash
scripts/download_deepseek_q4_gguf.sh
```

짧은 JSON 생성 테스트를 실행합니다.

```bash
PROMPT='Return JSON only: {"status":"ok"}' TOKENS=8 \
  scripts/run_deepseek_q4_gguf_demo.sh
```

짧은 일반 문장 생성도 확인할 수 있습니다.

```bash
PROMPT='Write one short sentence about local AI inference.' TOKENS=24 \
  scripts/run_deepseek_q4_gguf_demo.sh
```

## 성능 비교를 다시 돌리는 법

디코드 최적화 전/후 비교를 다시 실행하려면 다음 명령을 사용합니다.

```bash
scripts/run_deepseek_q4_perf_matrix.py \
  --mode infer \
  --infer-cases no_prewarm_madvise_off,no_prewarm_madvise_on \
  --prompts decode_json_seed,decode_plain_seed \
  --tokens 24 \
  --context 1024 \
  --repeats 3
```

5시간짜리 장기 수집을 다시 실행하려면 다음 명령을 사용합니다.

```bash
scripts/start_deepseek_q4_longrun_5h.sh
```

실험 결과 요약은 여기에 있습니다.

- 메인 성능 보고서:
  [docs/deepseek-q4-performance-matrix.md](docs/deepseek-q4-performance-matrix.md)
- 5시간 벤치마크 요약:
  [docs/results/deepseek_q4_longrun_5h/README.md](docs/results/deepseek_q4_longrun_5h/README.md)
- 5시간 집계 CSV:
  [docs/results/deepseek_q4_longrun_5h/summary.csv](docs/results/deepseek_q4_longrun_5h/summary.csv)
- 디코드 baseline 추가 결과:
  [docs/results/deepseek_q4_decode_baseline/summary.md](docs/results/deepseek_q4_decode_baseline/summary.md)
- 영어 영상 스크립트와 녹화 런북:
  [docs/video/README.md](docs/video/README.md)

## `packed_experts_q4`는 무엇인가

`packed_experts_q4`는 MLX/safetensors Q4 모델에서 routed expert 가중치를
실험하기 쉽게 다시 묶은 로컬 생성물입니다.

다만 생성된 파일이 약 **137.06 GiB**라서 GitHub에 직접 올리지 않았습니다.
대신 이 저장소에는 다음을 포함했습니다.

- 생성 스크립트
- 생성 방법
- 로컬 생성물의 manifest
- 검증 방법

재생성 방법은 [docs/packed-experts-q4.md](docs/packed-experts-q4.md)에
정리되어 있습니다.

## 이 프로젝트에서 바꾼 핵심

최적화는 의외로 작습니다.

1. 150GB GGUF 모델을 `mmap` 기반으로 유지합니다.
2. CPU repack을 꺼서 모델이 메모리에 한 번 더 복사되지 않게 합니다.
3. 고정된 expert 전체를 무작정 미리 올리는 방식에 의존하지 않습니다.
4. MoE 라우팅이 active expert를 고른 뒤, 해당 expert 행렬 범위에
   `MADV_WILLNEED`를 호출합니다.

즉, 별도의 거대한 expert 캐시를 직접 만들기보다 macOS의 파일 캐시를 활용하고,
필요한 페이지를 OS가 조금 더 빨리 준비하도록 힌트를 주는 방식입니다.

## 중요한 관찰

- 초기 trace에서는 page-in/disk-read proxy 기준으로 **97.6%가 I/O-active**였습니다.
- prefill과 decode는 둘 다 I/O 영향을 받지만 병목 양상이 다릅니다.
- prefill은 넓은 범위의 layer/page를 한꺼번에 읽는 성격이 강했습니다.
- decode는 토큰마다 expert 선택이 바뀌면서 생기는 지연이 더 중요했습니다.
- static top-16 expert prewarm은 cold start에는 도움이 되었지만, 5시간 steady-state에서는 가장 좋은 방법이 아니었습니다.
- 측정상 가장 명확한 개선은 `MADV_WILLNEED`였습니다.

## 다른 머신에서는 어떻게 쓰나

64GB보다 메모리가 많은 머신에서는 전략을 조금 다르게 잡을 수 있습니다.

- 128GB unified memory, 예를 들어 DGX Spark급 머신:
  더 많은 expert 페이지를 prewarm하되 OS 여유 메모리를 남기는 방식이 유리할 수 있습니다.
- 192GB-256GB 이상:
  더 큰 expert hot set을 유지하고, decode 중심으로 튜닝할 수 있습니다.
- 별도 GPU 서버:
  CPU/NVMe/PCIe/GPU 메모리 이동이 병목이 될 수 있으므로 측정 항목을 다시 나눠야 합니다.

자세한 노트는
[docs/appendix-other-machines.md](docs/appendix-other-machines.md)에 있습니다.

## 이름 설명

`MoE-MADV`는 **Mixture-of-Experts + MADV_WILLNEED**의 줄임말입니다.

이 프로젝트의 주장은 단순히 "큰 모델을 돌렸다"가 아닙니다.
DeepSeek V4 Flash 같은 MoE 모델은 기존 dense 모델과 병목이 다르기 때문에,
로컬 실행에서는 expert page loading 자체를 최적화 대상으로 봐야 한다는 것입니다.

## 원본 프로젝트와 관계

이 작업은 [`danveloper/flash-moe`](https://github.com/danveloper/flash-moe)에서
출발했습니다. 원본 flash-moe의 README와 개발 맥락은
[CLAUDE.md](CLAUDE.md)에 보존되어 있습니다.

이 저장소는 DeepSeek V4 Flash Q4 / GGUF / `MADV_WILLNEED` 실험에 초점을 둡니다.

## Agent Work Mem

이 프로젝트는
[agent-work-mem](https://github.com/daystar7777/agent-work-mem)의 도움을 받아
진행했습니다.

이 실험은 한 번에 끝나는 작업이 아니었습니다. 모델 다운로드, 실패한 실행 경로,
런타임 패치, 벤치마크, 5시간 데이터 수집, 문서 정리까지 이어졌습니다.
`agent-work-mem`은 어떤 모델을 받았는지, 어떤 시도가 실패했는지, 어떤 결과를
믿고 문서화해도 되는지 같은 작업 기억을 유지하는 데 도움이 되었습니다.
