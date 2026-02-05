const { LlamaModel, LlamaContext, setQuiet } = require('..')

// Parse command line arguments
const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
const modelPath = args[0] || './model.gguf'

// Suppress llama.cpp output for cleaner results
setQuiet(true)

console.log('Loading embedding model:', modelPath)
const model = new LlamaModel(modelPath, {
  nGpuLayers: 99
})

console.log('Embedding dimension:', model.embeddingDimension)

// Helper function to get embedding for a text string
// Creates a fresh context for each embedding to avoid state issues
function embed (text) {
  const ctx = new LlamaContext(model, {
    contextSize: 512,
    embeddings: true,
    poolingType: 2 // Mean pooling (0=unspecified, 1=none, 2=mean, 3=cls, 4=last)
  })
  
  // Tokenize the text (add BOS token)
  const tokens = model.tokenize(text, true)
  
  // Decode tokens to compute embeddings
  ctx.decode(tokens)
  
  // Get the pooled embedding vector (-1 = last/pooled output)
  const embedding = ctx.getEmbeddings(-1)
  
  // Copy the embedding since context will be freed
  const result = new Float32Array(embedding)
  
  ctx.free()
  return result
}

// Helper function to compute cosine similarity between two vectors
function cosineSimilarity (a, b) {
  let dotProduct = 0
  let normA = 0
  let normB = 0
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB))
}

// Example texts to embed
const texts = [
  'The cat sat on the mat.',
  'A feline rested on the rug.',
  'The dog ran in the park.',
  'Machine learning is a subset of artificial intelligence.'
]

console.log('\nComputing embeddings for sample texts...\n')

// Compute embeddings for all texts
const embeddings = texts.map((text, i) => {
  const vec = embed(text)
  console.log(`[${i + 1}] "${text}"`)
  console.log(`    First 5 values: [${Array.from(vec.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}...]`)
  return vec
})

// Compute similarity matrix
console.log('\nCosine similarity matrix:')
console.log('    ' + texts.map((_, i) => `[${i + 1}]`.padStart(6)).join(' '))

for (let i = 0; i < texts.length; i++) {
  const row = texts.map((_, j) => {
    const sim = cosineSimilarity(embeddings[i], embeddings[j])
    return sim.toFixed(3).padStart(6)
  }).join(' ')
  console.log(`[${i + 1}] ${row}`)
}

// Interactive: embed a custom string if provided
if (args[1]) {
  const customText = args.slice(1).join(' ')
  console.log(`\nCustom text: "${customText}"`)
  const customVec = embed(customText)
  console.log(`Embedding (first 10 values): [${Array.from(customVec.slice(0, 10)).map(v => v.toFixed(4)).join(', ')}...]`)
  
  console.log('\nSimilarity to sample texts:')
  texts.forEach((text, i) => {
    const sim = cosineSimilarity(customVec, embeddings[i])
    console.log(`  ${sim.toFixed(4)} - "${text}"`)
  })
}

// Cleanup
model.free()

