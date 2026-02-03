const path = require('bare-path')
const binding = require.addon(path.join(import.meta.dirname, '.'))

class LlamaModel {
  constructor (path, opts = {}) {
    this._handle = binding.loadModel(path, opts)
  }

  tokenize (text, addBos = true) {
    return binding.tokenize(this._handle, text, addBos)
  }

  detokenize (tokens) {
    return binding.detokenize(this._handle, tokens)
  }

  isEogToken (token) {
    return binding.isEogToken(this._handle, token)
  }

  get embeddingDimension () {
    return binding.getEmbeddingDimension(this._handle)
  }

  get trainingContextSize () {
    return binding.getTrainingContextSize(this._handle)
  }

  getMeta (key) {
    return binding.getModelMeta(this._handle, key)
  }

  get name () {
    return this.getMeta('general.name')
  }

  free () {
    if (this._handle) {
      binding.freeModel(this._handle)
      this._handle = null
    }
  }
}

class LlamaContext {
  constructor (model, opts = {}) {
    if (!(model instanceof LlamaModel)) {
      throw new Error('First argument must be a LlamaModel')
    }
    this._model = model
    this._handle = binding.createContext(model._handle, opts)
  }

  get contextSize () {
    return binding.getContextSize(this._handle)
  }

  decode (tokens) {
    binding.decode(this._handle, tokens)
  }

  getEmbeddings (idx = -1) {
    return binding.getEmbeddings(this._handle, idx)
  }

  clearMemory () {
    binding.clearMemory(this._handle)
  }

  free () {
    if (this._handle) {
      binding.freeContext(this._handle)
      this._handle = null
    }
  }
}

class LlamaSampler {
  constructor (model, opts = {}) {
    if (!(model instanceof LlamaModel)) {
      throw new Error('First argument must be a LlamaModel')
    }
    this._handle = binding.createSampler(model._handle, opts)
  }

  sample (ctx, idx = -1) {
    if (!(ctx instanceof LlamaContext)) {
      throw new Error('First argument must be a LlamaContext')
    }
    return binding.sample(ctx._handle, this._handle, idx)
  }

  accept (token) {
    binding.acceptToken(this._handle, token)
  }

  free () {
    if (this._handle) {
      binding.freeSampler(this._handle)
      this._handle = null
    }
  }
}

function generate (model, ctx, sampler, prompt, maxTokens = 128) {
  const tokens = model.tokenize(prompt, true)
  ctx.decode(tokens)

  const generated = []
  for (let i = 0; i < maxTokens; i++) {
    const token = sampler.sample(ctx, -1)
    if (model.isEogToken(token)) break

    sampler.accept(token)
    generated.push(token)

    // Decode single token for next iteration
    ctx.decode(new Int32Array([token]))
  }

  return model.detokenize(new Int32Array(generated))
}

// Log level: 0=off, 1=errors only, 2=all (default)
function setLogLevel (level) {
  binding.setLogLevel(level)
}

// Convenience function to suppress all llama.cpp output
function setQuiet (quiet = true) {
  binding.setLogLevel(quiet ? 0 : 2)
}

// Read GGUF metadata without loading the full model
function readGgufMeta (path, key) {
  return binding.readGgufMeta(path, key)
}

// Get model name from GGUF file
function getModelName (path) {
  return readGgufMeta(path, 'general.name')
}

module.exports = {
  LlamaModel,
  LlamaContext,
  LlamaSampler,
  generate,
  setLogLevel,
  setQuiet,
  readGgufMeta,
  getModelName,
  binding
}
