const { LlamaModel, setQuiet, systemInfo } = require('..')
const { getModelInfo } = require('../lib/ollama-models')
const os = require('bare-os')
const fs = require('bare-fs')
const path = require('bare-path')
const { spawnSync } = require('bare-subprocess')
const runGenerationBench = require('./generation')
const runEmbeddingsBench = require('./embeddings')

setQuiet(true)

const GENERATION_MODEL = 'llama3.2:1b'
const EMBEDDING_MODEL = 'nomic-embed-text'

function resolveModel (name) {
  const info = getModelInfo(name)
  if (!info) return null
  return info.path
}

function getLlamaVersion () {
  try {
    const result = spawnSync('git', ['-C', 'vendor/llama.cpp', 'describe', '--tags'])
    if (result.status === 0) return result.stdout.toString().trim()
    return 'unknown'
  } catch {
    return 'unknown'
  }
}

function getMetadata () {
  return {
    date: new Date().toISOString(),
    llamaVersion: getLlamaVersion(),
    systemInfo: systemInfo(),
    platform: os.platform(),
    arch: os.arch(),
    hostname: os.hostname()
  }
}

function saveResult (name, result) {
  const resultsDir = path.join(__dirname, 'results')
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir, { recursive: true })
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
  const filename = `${name}-${timestamp}.json`
  fs.writeFileSync(
    path.join(resultsDir, filename),
    JSON.stringify(result, null, 2)
  )

  // Append to history
  const historyPath = path.join(resultsDir, `${name}-history.jsonl`)
  fs.appendFileSync(historyPath, JSON.stringify(result) + '\n')

  return filename
}

console.log('# bare-llama.cpp Benchmark\n')

const metadata = getMetadata()
console.log('Date:', metadata.date)
console.log('llama.cpp:', metadata.llamaVersion)
console.log('Platform:', metadata.platform, metadata.arch)
console.log('System:', metadata.systemInfo)
console.log('')

// Generation benchmark
const genModelPath = resolveModel(GENERATION_MODEL)
if (genModelPath) {
  console.log('Loading generation model:', GENERATION_MODEL)
  const loadStart = Date.now()
  const model = new LlamaModel(genModelPath, { nGpuLayers: 99 })
  const loadTime = Date.now() - loadStart

  console.log(`Model loaded in ${loadTime} ms`)
  console.log('Running generation benchmark...')

  const genResult = runGenerationBench(model)
  model.free()

  const result = { ...metadata, model: GENERATION_MODEL, loadTimeMs: loadTime, ...genResult }
  const filename = saveResult('generation', result)

  console.log(`\nGeneration Results:`)
  console.log(`  Prompt processing: ${genResult.promptSpeed.toFixed(1)} tok/s (${genResult.promptTokens} tokens in ${genResult.promptTimeMs} ms)`)
  console.log(`  Generation speed:  ${genResult.genSpeed.toFixed(1)} tok/s (${genResult.generatedTokens} tokens in ${genResult.genTimeMs} ms)`)
  console.log(`  Time to first token: ${genResult.firstTokenMs} ms`)
  console.log(`  Saved: ${filename}`)
} else {
  console.log('Generation model not available:', GENERATION_MODEL)
}

console.log('')

// Embedding benchmark
const embModelPath = resolveModel(EMBEDDING_MODEL)
if (embModelPath) {
  console.log('Loading embedding model:', EMBEDDING_MODEL)
  const loadStart = Date.now()
  const model = new LlamaModel(embModelPath, { nGpuLayers: 99 })
  const loadTime = Date.now() - loadStart

  console.log(`Model loaded in ${loadTime} ms`)
  console.log('Running embedding benchmark (100 texts)...')

  const embResult = runEmbeddingsBench(model, 100)
  model.free()

  const result = { ...metadata, model: EMBEDDING_MODEL, loadTimeMs: loadTime, ...embResult }
  const filename = saveResult('embeddings', result)

  console.log(`\nEmbedding Results:`)
  console.log(`  New context:    ${embResult.newContextRate.toFixed(1)} emb/s`)
  console.log(`  Reuse context:  ${embResult.reuseContextRate.toFixed(1)} emb/s`)
  console.log(`  Speedup:        ${embResult.speedup.toFixed(2)}x`)
  console.log(`  Per embedding:  ${embResult.perEmbeddingMs.toFixed(2)} ms`)
  console.log(`  Saved: ${filename}`)
} else {
  console.log('Embedding model not available:', EMBEDDING_MODEL)
}

console.log('\nDone.')
