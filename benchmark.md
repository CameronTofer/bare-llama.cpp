
> bare-llama@0.1.0 test
> bare examples/regression-test.js

# bare-llama Regression Test

Date: 2026-01-29T23:56:48.532Z

Loading generation model: llama3.2
Loading embedding model: nomic-embed-text

Running generation tests...
Running embedding tests...

## Configuration

| Setting | Value |
|---------|-------|
| Generation Model | llama3.2 |
| Embedding Model | nomic-embed-text |
| Benchmark Iterations | 50 |
| Max Generate Tokens | 64 |

## Results

| Category | Test | Status | Details |
|----------|------|--------|---------|
| Setup | load-generation-model | PASS | 314.0 ms |
| Setup | load-embedding-model | PASS | 30.0 ms |
| Tokenization | roundtrip | PASS | 0.000 ms |
| Generation | basic | PASS | 125 tok/s, TTFT: 33.0 ms |
| Constraints | json-schema | PASS | 168.0 ms |
| Constraints | lark-grammar | PASS | 41.0 ms |
| Embeddings | semantic | PASS | 13.0 ms, dim: 768 |
| Embeddings | clearMemory | PASS | 249 emb/s, 1.51x |

**Summary:** 8 passed, 0 failed

## Performance Summary

| Metric | Value |
|--------|-------|
| Generation Speed | 124.8 tok/s |
| Time to First Token | 33.0 ms |
| Prompt Processing | 5.0 ms (5 tokens) |
| Embedding Speed (reuse) | 248.8 emb/s |
| Context Reuse Speedup | 1.51x |
| Load Time (llama3.2) | 314.0 ms |
| Load Time (nomic-embed-text) | 30.0 ms |

