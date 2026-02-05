# bare-llama

Native [llama.cpp](https://github.com/ggerganov/llama.cpp) bindings for [Bare](https://github.com/holepunchto/bare).

Run LLM inference directly in your Bare JavaScript applications with GPU acceleration support.

## Requirements

- CMake 3.25+
- C/C++ compiler (clang, gcc, or MSVC)
- Node.js (for npm/cmake-bare)
- Bare runtime

## Building

Clone with submodules:

```bash
git clone --recursive https://github.com/CameronTofer/bare-llama.cpp
cd bare-llama.cpp
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

Install dependencies and build:

```bash
npm install
```

Or manually:

```bash
bare-make generate
bare-make build
bare-make install
```

This creates `prebuilds/<platform>-<arch>/bare-llama.bare`.

### Build Options

For a debug build:

```bash
bare-make generate -- -D CMAKE_BUILD_TYPE=Debug
bare-make build
```

To disable GPU acceleration:

```bash
bare-make generate -- -D GGML_METAL=OFF -D GGML_CUDA=OFF
bare-make build
```

## Usage

```javascript
const { LlamaModel, LlamaContext, LlamaSampler, generate } = require('bare-llama')

// Load model (GGUF format)
const model = new LlamaModel('./model.gguf', {
  nGpuLayers: 99  // Offload layers to GPU (0 = CPU only)
})

// Create context
const ctx = new LlamaContext(model, {
  contextSize: 2048,  // Max context length
  batchSize: 512      // Batch size for prompt processing
})

// Create sampler
const sampler = new LlamaSampler(model, {
  temp: 0.7,    // Temperature (0 = greedy)
  topK: 40,     // Top-K sampling
  topP: 0.95    // Top-P (nucleus) sampling
})

// Generate text
const output = generate(model, ctx, sampler, 'The meaning of life is', 128)
console.log(output)

// Cleanup
sampler.free()
ctx.free()
model.free()
```

### Embeddings

```javascript
const { LlamaModel, LlamaContext, setQuiet } = require('bare-llama')

setQuiet(true)

const model = new LlamaModel('./embedding-model.gguf', { nGpuLayers: 99 })
const ctx = new LlamaContext(model, {
  contextSize: 512,
  embeddings: true,
  poolingType: 2  // 0=unspecified, 1=none, 2=mean, 3=cls, 4=last
})

const tokens = model.tokenize('Hello world', true)
ctx.decode(tokens)
const embedding = ctx.getEmbeddings(-1)  // Float32Array

// Reuse context for multiple embeddings
ctx.clearMemory()
const tokens2 = model.tokenize('Another text', true)
ctx.decode(tokens2)
const embedding2 = ctx.getEmbeddings(-1)

ctx.free()
model.free()
```

### Constrained Generation

```javascript
const { LlamaModel, LlamaContext, LlamaSampler, generate, setQuiet } = require('bare-llama')

setQuiet(true)

const model = new LlamaModel('./model.gguf', { nGpuLayers: 99 })
const ctx = new LlamaContext(model, { contextSize: 2048 })

// JSON schema constraint (requires llguidance)
const schema = JSON.stringify({
  type: 'object',
  properties: { name: { type: 'string' }, age: { type: 'integer' } },
  required: ['name', 'age']
})
const sampler = new LlamaSampler(model, { temp: 0, json: schema })

// Lark grammar constraint
const sampler2 = new LlamaSampler(model, { temp: 0, lark: 'start: "yes" | "no"' })
```

## Examples

| Example | Description |
|---------|-------------|
| `examples/text-generation.js` | High-level `generate()` API |
| `examples/token-by-token.js` | Manual tokenize/sample/decode loop |
| `examples/cosine-similarity.js` | Embeddings + semantic similarity |
| `examples/json-constrained-output.js` | JSON schema constrained generation |
| `examples/lark-constrained-output.js` | Lark grammar constrained generation |
| `examples/tool-use-agent.js` | Agentic tool calling with multi-turn |

Run examples with:

```bash
bare examples/text-generation.js -- /path/to/model.gguf
```

## Testing

Tests use [brittle](https://github.com/holepunchto/brittle) and skip gracefully when models aren't available.

```bash
npm test
```

Model-dependent tests require [Ollama](https://ollama.com) models installed locally:

```bash
ollama pull llama3.2:1b        # generation tests
ollama pull nomic-embed-text   # embedding tests
```

## Benchmarks

```bash
npm run bench
```

Results are saved to `bench/results/` as JSON with full metadata (llama.cpp version, system info, platform). History is tracked in JSONL files for comparison across runs.

## API Reference

### LlamaModel

```javascript
new LlamaModel(path, options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nGpuLayers` | number | 0 | Number of layers to offload to GPU |

**Properties:**

- `name` - Model name from metadata
- `embeddingDimension` - Embedding vector size
- `trainingContextSize` - Training context length

**Methods:**

- `tokenize(text, addBos?)` - Convert text to tokens (Int32Array)
- `detokenize(tokens)` - Convert tokens back to text
- `isEogToken(token)` - Check if token is end-of-generation
- `getMeta(key)` - Get model metadata by key
- `free()` - Release model resources

### LlamaContext

```javascript
new LlamaContext(model, options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `contextSize` | number | 512 | Maximum context length |
| `batchSize` | number | 512 | Batch size for processing |
| `embeddings` | boolean | false | Enable embedding mode |
| `poolingType` | number | 0 | Pooling strategy (0=unspecified, 1=none, 2=mean, 3=cls, 4=last) |

**Properties:**

- `contextSize` - Actual context size

**Methods:**

- `decode(tokens)` - Process tokens through the model
- `getEmbeddings(idx)` - Get embedding vector (Float32Array)
- `clearMemory()` - Clear context for reuse (faster than creating new context)
- `free()` - Release context resources

### LlamaSampler

```javascript
new LlamaSampler(model, options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temp` | number | 0 | Temperature (0 = greedy sampling) |
| `topK` | number | 40 | Top-K sampling parameter |
| `topP` | number | 0.95 | Top-P (nucleus) sampling parameter |
| `json` | string | - | JSON schema constraint (requires llguidance) |
| `lark` | string | - | Lark grammar constraint (requires llguidance) |

**Methods:**

- `sample(ctx, idx)` - Sample next token (-1 for last position)
- `accept(token)` - Accept token into sampler state
- `free()` - Release sampler resources

### generate()

```javascript
generate(model, ctx, sampler, prompt, maxTokens?)
```

Convenience function for simple text generation. Returns the generated text (not including the prompt).

### Utility Functions

- `setQuiet(quiet?)` - Suppress llama.cpp output
- `setLogLevel(level)` - Set log level (0=off, 1=errors, 2=all)
- `readGgufMeta(path, key)` - Read GGUF metadata without loading the model
- `getModelName(path)` - Get model name from GGUF file
- `systemInfo()` - Get hardware/instruction set info (AVX, NEON, Metal, CUDA)

## Project Structure

```
index.js              Main module
binding.cpp           C++ native bindings
lib/
  ollama-models.js    Ollama model discovery
  ollama.js           GGUF metadata + Jinja chat templates
test/                 Brittle test suite
bench/                Benchmark system
examples/             Usage examples
tools/
  ollama-hyperdrive.js  P2P model distribution (standalone CLI)
```

## Models

This addon works with GGUF format models. You can use models from [Ollama](https://ollama.com) (auto-detected from `~/.ollama/models`) or download GGUF files directly from [Hugging Face](https://huggingface.co/models?search=gguf).

## Platform Support

| Platform | Architecture | GPU Support |
|----------|--------------|-------------|
| macOS | arm64, x64 | Metal |
| Linux | x64, arm64 | CUDA (if available) |
| Windows | x64 | CUDA (if available) |

## License

MIT
