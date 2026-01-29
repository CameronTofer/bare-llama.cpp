const { LlamaModel, LlamaContext, LlamaSampler, generate, setQuiet } = require('..')
const { getModelPath, getModelInfo } = require('./ollama-models')
const config = require('./test-config')

setQuiet(true)

// ============================================================================
// Helpers
// ============================================================================

function resolveModel (nameOrPath) {
  if (nameOrPath.endsWith('.gguf') || nameOrPath.startsWith('/') || nameOrPath.startsWith('./')) {
    return nameOrPath
  }
  const info = getModelInfo(nameOrPath)
  if (!info) throw new Error(`Model not found: ${nameOrPath}`)
  return info.path
}

function formatMs (ms) {
  if (ms < 1) return ms.toFixed(3) + ' ms'
  if (ms < 1000) return ms.toFixed(1) + ' ms'
  return (ms / 1000).toFixed(2) + ' s'
}

function formatRate (count, ms, unit = '/sec') {
  const rate = count / ms * 1000
  if (rate < 1) return rate.toFixed(3) + unit
  if (rate < 100) return rate.toFixed(1) + unit
  return Math.round(rate) + unit
}

function cosineSimilarity (a, b) {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

// ============================================================================
// Test Results Storage
// ============================================================================

const results = []

function record (category, test, passed, details = {}) {
  results.push({ category, test, passed, ...details })
}

// ============================================================================
// Tests
// ============================================================================

function testTokenization (model) {
  const testCases = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    '日本語テスト',
    'emoji: 🎉🚀',
    ''
  ]

  let allPassed = true
  const start = Date.now()

  for (const text of testCases) {
    const tokens = model.tokenize(text, false)
    const decoded = model.detokenize(tokens)
    if (decoded !== text) {
      allPassed = false
    }
  }

  const elapsed = Date.now() - start
  record('Tokenization', 'roundtrip', allPassed, { time: elapsed })
}

function testGeneration (model) {
  const ctx = new LlamaContext(model, { contextSize: 2048, batchSize: 512 })
  const sampler = new LlamaSampler(model, { temp: 0.7, topK: 40, topP: 0.95 })

  const prompt = 'The quick brown fox'
  const tokens = model.tokenize(prompt, true)

  // Measure prompt processing
  const decodeStart = Date.now()
  ctx.decode(tokens)
  const promptTime = Date.now() - decodeStart

  // Measure generation
  const genStart = Date.now()
  const generated = []
  let firstTokenTime = null

  for (let i = 0; i < config.maxGenerateTokens; i++) {
    const token = sampler.sample(ctx, -1)

    if (firstTokenTime === null) {
      firstTokenTime = Date.now() - genStart
    }

    if (model.isEogToken(token)) break

    sampler.accept(token)
    generated.push(token)
    ctx.decode(new Int32Array([token]))
  }

  const genTime = Date.now() - genStart
  const output = model.detokenize(new Int32Array(generated))
  const passed = generated.length > 0 && output.length > 0

  sampler.free()
  ctx.free()

  record('Generation', 'basic', passed, {
    promptTokens: tokens.length,
    promptTime,
    generatedTokens: generated.length,
    genTime,
    tokensPerSec: generated.length / genTime * 1000,
    firstTokenTime
  })
}

function testJsonSchema (model) {
  const ctx = new LlamaContext(model, { contextSize: 2048 })

  const schema = JSON.stringify({
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'integer' }
    },
    required: ['name', 'age'],
    additionalProperties: false
  })

  let sampler
  try {
    sampler = new LlamaSampler(model, { temp: 0, json: schema })
  } catch (e) {
    record('Constraints', 'json-schema', false, { error: 'llguidance not available' })
    ctx.free()
    return
  }

  const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate JSON for a person named Alice who is 30.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

  const start = Date.now()
  const output = generate(model, ctx, sampler, prompt, 64)
  const elapsed = Date.now() - start

  let passed = false
  let parsed = null
  try {
    parsed = JSON.parse(output.trim())
    passed = typeof parsed.name === 'string' && typeof parsed.age === 'number'
  } catch (e) {
    passed = false
  }

  sampler.free()
  ctx.free()

  record('Constraints', 'json-schema', passed, {
    time: elapsed,
    output: output.trim().slice(0, 100),
    error: passed ? null : 'invalid JSON or missing fields'
  })
}

function testLarkGrammar (model) {
  const ctx = new LlamaContext(model, { contextSize: 2048 })

  const grammar = `
start: RESPONSE
RESPONSE: "yes" | "no"
`

  let sampler
  try {
    sampler = new LlamaSampler(model, { temp: 0, lark: grammar })
  } catch (e) {
    record('Constraints', 'lark-grammar', false, { error: 'llguidance not available' })
    ctx.free()
    return
  }

  const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIs the sky blue? Answer yes or no only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'

  const start = Date.now()
  const output = generate(model, ctx, sampler, prompt, 4)
  const elapsed = Date.now() - start

  const trimmed = output.trim()
  const passed = trimmed === 'yes' || trimmed === 'no'

  sampler.free()
  ctx.free()

  record('Constraints', 'lark-grammar', passed, {
    time: elapsed,
    output: trimmed,
    error: passed ? null : `expected "yes" or "no", got "${trimmed}"`
  })
}

function testEmbeddings (model) {
  const ctx = new LlamaContext(model, {
    contextSize: 512,
    embeddings: true,
    poolingType: 2
  })

  const texts = [
    'The cat sat on the mat.',
    'A feline rested on the rug.',
    'Machine learning is a subset of artificial intelligence.'
  ]

  const embeddings = []
  const start = Date.now()

  for (const text of texts) {
    ctx.clearMemory()
    const tokens = model.tokenize(text, true)
    ctx.decode(tokens)
    embeddings.push(new Float32Array(ctx.getEmbeddings(-1)))
  }

  const elapsed = Date.now() - start

  // Semantic similarity check: cat/feline should be more similar than cat/ML
  const simCatFeline = cosineSimilarity(embeddings[0], embeddings[1])
  const simCatML = cosineSimilarity(embeddings[0], embeddings[2])
  const passed = simCatFeline > simCatML

  ctx.free()

  record('Embeddings', 'semantic', passed, {
    time: elapsed,
    dimension: embeddings[0].length,
    simCatFeline: simCatFeline.toFixed(4),
    simCatML: simCatML.toFixed(4)
  })
}

function testClearMemory (model) {
  const texts = []
  const phrases = [
    'The quick brown fox jumps over the lazy dog.',
    'Machine learning models can process natural language.',
    'Artificial intelligence is transforming many industries.'
  ]
  for (let i = 0; i < config.benchmarkIterations; i++) {
    texts.push(phrases[i % phrases.length])
  }

  // Method 1: New context per embedding
  const start1 = Date.now()
  for (const text of texts) {
    const ctx = new LlamaContext(model, {
      contextSize: 512,
      embeddings: true,
      poolingType: 2
    })
    const tokens = model.tokenize(text, true)
    ctx.decode(tokens)
    ctx.getEmbeddings(-1)
    ctx.free()
  }
  const time1 = Date.now() - start1

  // Method 2: Reuse context with clearMemory
  const ctx = new LlamaContext(model, {
    contextSize: 512,
    embeddings: true,
    poolingType: 2
  })

  const start2 = Date.now()
  for (const text of texts) {
    ctx.clearMemory()
    const tokens = model.tokenize(text, true)
    ctx.decode(tokens)
    ctx.getEmbeddings(-1)
  }
  const time2 = Date.now() - start2
  ctx.free()

  const speedup = time1 / time2
  const passed = speedup > 1.0 // Should be faster with reuse

  record('Embeddings', 'clearMemory', passed, {
    iterations: config.benchmarkIterations,
    newContextTime: time1,
    reuseContextTime: time2,
    speedup: speedup.toFixed(2) + 'x',
    reuseRate: texts.length / time2 * 1000
  })
}

// ============================================================================
// Main
// ============================================================================

console.log('# bare-llama Regression Test\n')
console.log('Date:', new Date().toISOString())
console.log('')

// Load generation model
let genModel = null
let genModelPath = null
try {
  genModelPath = resolveModel(config.generationModel)
  console.log('Loading generation model:', config.generationModel)
  const loadStart = Date.now()
  genModel = new LlamaModel(genModelPath, { nGpuLayers: 99 })
  const loadTime = Date.now() - loadStart
  record('Setup', 'load-generation-model', true, { time: loadTime, model: config.generationModel })
} catch (e) {
  console.log('Failed to load generation model:', e.message)
  record('Setup', 'load-generation-model', false, { error: e.message, model: config.generationModel })
}

// Load embedding model
let embModel = null
let embModelPath = null
try {
  embModelPath = resolveModel(config.embeddingModel)
  console.log('Loading embedding model:', config.embeddingModel)
  const loadStart = Date.now()
  embModel = new LlamaModel(embModelPath, { nGpuLayers: 99 })
  const loadTime = Date.now() - loadStart
  record('Setup', 'load-embedding-model', true, { time: loadTime, model: config.embeddingModel })
} catch (e) {
  console.log('Failed to load embedding model:', e.message)
  record('Setup', 'load-embedding-model', false, { error: e.message, model: config.embeddingModel })
}

console.log('')

// Run tests
if (genModel) {
  console.log('Running generation tests...')
  testTokenization(genModel)
  testGeneration(genModel)
  testJsonSchema(genModel)
  testLarkGrammar(genModel)
}

if (embModel) {
  console.log('Running embedding tests...')
  testEmbeddings(embModel)
  testClearMemory(embModel)
}

// Cleanup
if (genModel) genModel.free()
if (embModel) embModel.free()

// ============================================================================
// Output Results
// ============================================================================

console.log('\n## Configuration\n')
console.log('| Setting | Value |')
console.log('|---------|-------|')
console.log(`| Generation Model | ${config.generationModel} |`)
console.log(`| Embedding Model | ${config.embeddingModel} |`)
console.log(`| Benchmark Iterations | ${config.benchmarkIterations} |`)
console.log(`| Max Generate Tokens | ${config.maxGenerateTokens} |`)

console.log('\n## Results\n')
console.log('| Category | Test | Status | Details |')
console.log('|----------|------|--------|---------|')

for (const r of results) {
  const status = r.passed ? 'PASS' : 'FAIL'
  const details = []

  if (r.time !== undefined) details.push(formatMs(r.time))
  if (r.tokensPerSec !== undefined) details.push(formatRate(r.tokensPerSec, 1000, ' tok/s'))
  if (r.reuseRate !== undefined) details.push(formatRate(r.reuseRate, 1000, ' emb/s'))
  if (r.speedup !== undefined) details.push(r.speedup)
  if (r.firstTokenTime !== undefined) details.push('TTFT: ' + formatMs(r.firstTokenTime))
  if (r.dimension !== undefined) details.push('dim: ' + r.dimension)
  if (r.output !== undefined && !r.passed) details.push('got: ' + r.output)
  if (r.error !== undefined && r.error !== null) details.push('error: ' + r.error)

  console.log(`| ${r.category} | ${r.test} | ${status} | ${details.join(', ')} |`)
}

// Summary
const passed = results.filter(r => r.passed).length
const failed = results.filter(r => !r.passed).length
console.log('')
console.log(`**Summary:** ${passed} passed, ${failed} failed`)

if (failed > 0) {
  console.log('\n### Failed Tests\n')
  for (const r of results.filter(r => !r.passed)) {
    console.log(`- ${r.category}/${r.test}: ${r.error || 'unexpected result'}`)
  }
}

// Performance summary
console.log('\n## Performance Summary\n')
console.log('| Metric | Value |')
console.log('|--------|-------|')

const genResult = results.find(r => r.test === 'basic' && r.category === 'Generation')
if (genResult && genResult.passed) {
  console.log(`| Generation Speed | ${genResult.tokensPerSec.toFixed(1)} tok/s |`)
  console.log(`| Time to First Token | ${formatMs(genResult.firstTokenTime)} |`)
  console.log(`| Prompt Processing | ${formatMs(genResult.promptTime)} (${genResult.promptTokens} tokens) |`)
}

const embResult = results.find(r => r.test === 'clearMemory')
if (embResult && embResult.passed) {
  console.log(`| Embedding Speed (reuse) | ${embResult.reuseRate.toFixed(1)} emb/s |`)
  console.log(`| Context Reuse Speedup | ${embResult.speedup} |`)
}

const loadGen = results.find(r => r.test === 'load-generation-model')
const loadEmb = results.find(r => r.test === 'load-embedding-model')
if (loadGen && loadGen.passed) {
  console.log(`| Load Time (${config.generationModel}) | ${formatMs(loadGen.time)} |`)
}
if (loadEmb && loadEmb.passed) {
  console.log(`| Load Time (${config.embeddingModel}) | ${formatMs(loadEmb.time)} |`)
}

console.log('')
