# RAG Experiment: LlamaIndex vs LangChain

LlamaIndex와 LangChain 두 프레임워크에서 6가지 유사도 알고리즘의 성능을 비교하는 RAG(Retrieval-Augmented Generation) 실험 프로젝트입니다.  
LLM과 임베딩 모델 모두 **Ollama 로컬 실행**으로 API 비용 없이 무료로 동작합니다.

---

## 실험 구성

| 항목 | 내용 |
|------|------|
| LLM | `gemma4:26b` via Ollama |
| 임베딩 | `nomic-embed-text` via Ollama (768-dim) |
| 프레임워크 | LlamaIndex, LangChain |
| 유사도 방법 | cosine, euclidean, dot_product, manhattan, bm25, hybrid(RRF) |
| 데이터 | 삼성 냉장고 사용설명서 PDF (88페이지, 238 chunks) |
| 쿼리 | 가전제품 관련 한국어 질문 15개 |

---

## 프로젝트 구조

```
RAG_Experiment/
├── data/
│   └── pdfs/                  # 실험용 PDF 파일
├── experiments/
│   ├── run_experiments.py     # 메인 실험 실행 스크립트
│   └── sample_queries.py      # 테스트 쿼리 15개
├── src/
│   ├── config.py              # 실험 설정 (모델, 청킹, top-k 등)
│   ├── data_loader.py         # PDF 로드 및 청킹
│   ├── embeddings.py          # Ollama 임베딩 래퍼
│   ├── similarity_search.py   # 6가지 유사도 검색 구현
│   ├── llamaindex_rag.py      # LlamaIndex RAG 파이프라인
│   ├── langchain_rag.py       # LangChain RAG 파이프라인
│   ├── evaluator.py           # 결과 평가 (컨텍스트 관련도 등)
│   └── visualizer.py          # 결과 시각화
├── results/
│   ├── plots/                 # 생성된 비교 그래프
│   └── raw/                   # 실험 결과 JSON
├── notebooks/
│   └── analysis.ipynb         # 결과 분석 노트북
├── .env.example               # 환경 변수 예시
└── requirements.txt
```

---

## 설치 및 실행

### 1. Ollama 설치 및 모델 다운로드

[ollama.com](https://ollama.com/download/mac) 에서 설치 후:

```bash
ollama pull nomic-embed-text   # 임베딩 모델 (~274MB)
ollama pull gemma4:26b         # LLM (~16GB)
```

### 2. Python 환경 세팅

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. 환경 변수 설정

```bash
cp .env.example .env
```

### 4. 실험 실행

```bash
# 전체 실험 (LlamaIndex + LangChain, 6가지 유사도)
python experiments/run_experiments.py

# 유사도 검색만 빠르게 (LLM 호출 없음)
python experiments/run_experiments.py --similarity-only

# 특정 프레임워크/방법만
python experiments/run_experiments.py --frameworks langchain --methods cosine bm25 hybrid

# 모델 직접 지정
python experiments/run_experiments.py --llm-provider ollama --llm-model gemma4:26b
```

---

## 유사도 알고리즘

| 방법 | 설명 |
|------|------|
| `cosine` | FAISS IndexFlatIP + L2 정규화 벡터 |
| `euclidean` | FAISS IndexFlatL2 |
| `dot_product` | FAISS IndexFlatIP (정규화 없음) |
| `manhattan` | sklearn L1 거리 |
| `bm25` | 희소 키워드 기반 BM25Okapi |
| `hybrid` | BM25 + cosine RRF(Reciprocal Rank Fusion) 융합 |

---

## 실험 결과 (LangChain 기준)

| 방법 | 평균 응답시간(s) | 컨텍스트 관련도 |
|------|:--------------:|:--------------:|
| cosine | 6.82 | **0.6245** |
| euclidean | 7.02 | **0.6245** |
| dot_product | 7.36 | **0.6245** |
| manhattan | 7.87 | 0.6232 |
| hybrid | 8.23 | 0.5949 |
| bm25 | **6.43** | 0.5072 |

> 컨텍스트 관련도: 검색된 청크와 쿼리 벡터 간 코사인 유사도 평균

---

## 출력 결과물

실험 완료 후 `results/` 디렉터리에 다음 파일이 생성됩니다:

- `results/plots/latency_comparison.png` — 프레임워크별 응답 시간 비교
- `results/plots/context_relevance.png` — 유사도 방법별 컨텍스트 관련도
- `results/plots/heatmap_avg_ctx_relevance.png` — 관련도 히트맵
- `results/plots/heatmap_total_time_s.png` — 응답 시간 히트맵
- `results/plots/summary.csv` — 전체 요약 테이블
- `results/raw/rag_results_*.json` — 쿼리별 원시 결과

---

## 환경 변수 (.env)

```env
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=gemma4:26b
EMBEDDING_MODEL=nomic-embed-text
```
