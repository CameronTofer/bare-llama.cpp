const { LlamaModel, setQuiet } = require('..')
const { getModelInfo } = require('../lib/ollama-models')

setQuiet(true)

const GENERATION_MODEL = 'llama3.2:1b'
const EMBEDDING_MODEL = 'nomic-embed-text'

function resolveModel (nameOrPath) {
  if (nameOrPath.endsWith('.gguf') || nameOrPath.startsWith('/') || nameOrPath.startsWith('./')) {
    return nameOrPath
  }
  const info = getModelInfo(nameOrPath)
  if (!info) return null
  return info.path
}

function tryLoadModel (name, opts = {}) {
  const modelPath = resolveModel(name)
  if (!modelPath) return null
  try {
    const model = new LlamaModel(modelPath, { nGpuLayers: 99, ...opts })
    return { model, modelPath }
  } catch {
    return null
  }
}

module.exports = {
  GENERATION_MODEL,
  EMBEDDING_MODEL,
  resolveModel,
  tryLoadModel
}
