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
git clone --recursive https://github.com/user/bare-llama.cpp
cd bare-llama.cpp
```

Or if already cloned:

```bash
git submodule update --init --recursive
```

Install dependencies and build:

```bash
npm install
npm run prebuild
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
const sampler = new LlamaSampler({
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

See `examples/` for more:
- `basic.js` - Simple generation using the high-level API
- `lowlevel.js` - Token-by-token generation with full control

Run examples with:
```bash
bare examples/basic.js -- /path/to/model.gguf
```

## API Reference

### LlamaModel

```javascript
new LlamaModel(path, options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `nGpuLayers` | number | 0 | Number of layers to offload to GPU |

**Methods:**

- `tokenize(text, addBos?)` - Convert text to tokens (Int32Array)
- `detokenize(tokens)` - Convert tokens back to text
- `isEogToken(token)` - Check if token is end-of-generation
- `free()` - Release model resources

### LlamaContext

```javascript
new LlamaContext(model, options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `contextSize` | number | 512 | Maximum context length |
| `batchSize` | number | 512 | Batch size for processing |

**Methods:**

- `decode(tokens)` - Process tokens through the model
- `free()` - Release context resources

### LlamaSampler

```javascript
new LlamaSampler(options?)
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `temp` | number | 0 | Temperature (0 = greedy sampling) |
| `topK` | number | 40 | Top-K sampling parameter |
| `topP` | number | 0.95 | Top-P (nucleus) sampling parameter |

**Methods:**

- `sample(ctx, idx)` - Sample next token (-1 for last position)
- `accept(token)` - Accept token into sampler state
- `free()` - Release sampler resources

### generate()

```javascript
generate(model, ctx, sampler, prompt, maxTokens?)
```

Convenience function for simple text generation. Returns the generated text (not including the prompt).

## Models

This addon works with GGUF format models. You can find models at:

- [Hugging Face](https://huggingface.co/models?search=gguf)
- [TheBloke's models](https://huggingface.co/TheBloke)

Example models to try:

- `TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf` - Small, fast
- `Mistral-7B-Instruct-v0.2.Q4_K_M.gguf` - Good quality
- `Llama-2-13B-chat.Q4_K_M.gguf` - Higher quality

## Platform Support

| Platform | Architecture | GPU Support |
|----------|--------------|-------------|
| macOS | arm64, x64 | Metal |
| Linux | x64, arm64 | CUDA (if available) |
| Windows | x64 | CUDA (if available) |

## License

MIT
