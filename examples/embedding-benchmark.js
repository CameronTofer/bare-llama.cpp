const { LlamaModel, LlamaContext, setQuiet } = require('..')
const { getModelPath, listModels, getModelInfo } = require('./ollama-models')

// Parse command line arguments
const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
const modelArg = args[0] || 'embeddinggemma'
const numTexts = parseInt(args[1]) || 500

setQuiet(true)

// Resolve model path - check if it's an Ollama model name or a file path
let modelPath
if (modelArg.endsWith('.gguf') || modelArg.startsWith('/') || modelArg.startsWith('./')) {
  modelPath = modelArg
} else {
  // Treat as Ollama model name
  const info = getModelInfo(modelArg)
  if (info) {
    console.log('Using Ollama model:', modelArg)
    console.log('  Size:', info.sizeHuman)
    modelPath = info.path
  } else {
    console.error('Failed to find Ollama model:', modelArg)
    console.log('\nAvailable Ollama models:')
    listModels().slice(0, 20).forEach(m => console.log(' -', m))
    throw new Error('Model not found')
  }
}

console.log('=== Embedding Benchmark ===\n')
console.log('Model path:', modelPath)
console.log('Number of texts:', numTexts)

// Generate sample texts of varying lengths
const samplePhrases = [
  'The quick brown fox jumps over the lazy dog.',
  'Machine learning models can process natural language.',
  'Artificial intelligence is transforming many industries.',
  'Deep learning requires large amounts of training data.',
  'Neural networks are inspired by biological brains.',
  'Natural language processing enables text understanding.',
  'Computer vision allows machines to interpret images.',
  'Reinforcement learning agents learn through trial and error.',
  'Transfer learning reduces the need for labeled data.',
  'Transformers have revolutionized NLP since 2017.',
  'Attention mechanisms help models focus on relevant parts.',
  'Embeddings represent words as dense vectors.',
  'Semantic similarity can be measured with cosine distance.',
  'Vector databases enable fast similarity search.',
  'RAG combines retrieval with generation for better answers.',
  'Fine-tuning adapts pre-trained models to specific tasks.',
  'Quantization reduces model size with minimal accuracy loss.',
  'Prompt engineering is crucial for effective LLM usage.',
  'Context windows limit how much text models can process.',
  'Tokenization breaks text into smaller units for processing.'
]

// Generate test texts
const texts = []
for (let i = 0; i < numTexts; i++) {
  // Pick 1-3 random phrases and combine them
  const numPhrases = 1 + Math.floor(Math.random() * 3)
  const selected = []
  for (let j = 0; j < numPhrases; j++) {
    selected.push(samplePhrases[Math.floor(Math.random() * samplePhrases.length)])
  }
  texts.push(selected.join(' '))
}

// Calculate average text length
const avgLength = texts.reduce((sum, t) => sum + t.length, 0) / texts.length
console.log('Average text length:', Math.round(avgLength), 'chars')

// Load model
console.log('\nLoading model...')
const loadStart = Date.now()
const model = new LlamaModel(modelPath, { nGpuLayers: 99 })
console.log('Model loaded in', Date.now() - loadStart, 'ms')
console.log('Embedding dimension:', model.embeddingDimension)

// ============================================
// Method 1: New context per embedding (naive)
// ============================================
console.log('\n--- Method 1: New context per embedding ---')

function embedNaive (text) {
  const ctx = new LlamaContext(model, {
    contextSize: 512,
    embeddings: true,
    poolingType: 2
  })
  const tokens = model.tokenize(text, true)
  ctx.decode(tokens)
  const result = new Float32Array(ctx.getEmbeddings(-1))
  ctx.free()
  return result
}

// Warm up
embedNaive(texts[0])

const start1 = Date.now()
const embeddings1 = []
for (let i = 0; i < texts.length; i++) {
  embeddings1.push(embedNaive(texts[i]))
  if ((i + 1) % 100 === 0) {
    console.log(`  Progress: ${i + 1}/${texts.length}`)
  }
}
const time1 = Date.now() - start1
console.log(`  Completed ${texts.length} embeddings in ${time1} ms`)
console.log(`  Rate: ${(texts.length / time1 * 1000).toFixed(1)} embeddings/sec`)
console.log(`  Per embedding: ${(time1 / texts.length).toFixed(2)} ms`)

// ============================================
// Method 2: Reuse context with clearMemory()
// ============================================
console.log('\n--- Method 2: Reuse context with clearMemory() ---')

const ctx = new LlamaContext(model, {
  contextSize: 512,
  embeddings: true,
  poolingType: 2
})

function embedReuse (text) {
  ctx.clearMemory()
  const tokens = model.tokenize(text, true)
  ctx.decode(tokens)
  return new Float32Array(ctx.getEmbeddings(-1))
}

// Warm up
embedReuse(texts[0])

const start2 = Date.now()
const embeddings2 = []
for (let i = 0; i < texts.length; i++) {
  embeddings2.push(embedReuse(texts[i]))
  if ((i + 1) % 100 === 0) {
    console.log(`  Progress: ${i + 1}/${texts.length}`)
  }
}
const time2 = Date.now() - start2
console.log(`  Completed ${texts.length} embeddings in ${time2} ms`)
console.log(`  Rate: ${(texts.length / time2 * 1000).toFixed(1)} embeddings/sec`)
console.log(`  Per embedding: ${(time2 / texts.length).toFixed(2)} ms`)

ctx.free()

// ============================================
// Results comparison
// ============================================
console.log('\n=== Results ===')
console.log(`Method 1 (new context):    ${time1} ms total`)
console.log(`Method 2 (reuse context):  ${time2} ms total`)
console.log(`Speedup: ${(time1 / time2).toFixed(2)}x faster`)

// Verify embeddings are identical
function cosineSimilarity (a, b) {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

// Check first few embeddings match
let allMatch = true
for (let i = 0; i < Math.min(10, texts.length); i++) {
  const sim = cosineSimilarity(embeddings1[i], embeddings2[i])
  if (sim < 0.9999) {
    console.log(`\nWarning: Embedding ${i} differs (similarity: ${sim.toFixed(6)})`)
    allMatch = false
  }
}

if (allMatch) {
  console.log('\n✓ Verified: Both methods produce identical embeddings')
}

// Cleanup
model.free()

