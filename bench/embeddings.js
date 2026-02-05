const { LlamaContext } = require('..')

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
  'Transformers have revolutionized NLP since 2017.'
]

module.exports = function runEmbeddingsBench (model, numTexts = 100) {
  const texts = []
  for (let i = 0; i < numTexts; i++) {
    const n = 1 + Math.floor(Math.random() * 3)
    const selected = []
    for (let j = 0; j < n; j++) {
      selected.push(samplePhrases[Math.floor(Math.random() * samplePhrases.length)])
    }
    texts.push(selected.join(' '))
  }

  // Method 1: New context per embedding
  const start1 = Date.now()
  for (const text of texts) {
    const ctx = new LlamaContext(model, { contextSize: 512, embeddings: true, poolingType: 2 })
    const tokens = model.tokenize(text, true)
    ctx.decode(tokens)
    ctx.getEmbeddings(-1)
    ctx.free()
  }
  const newCtxTime = Date.now() - start1

  // Method 2: Reuse context with clearMemory
  const ctx = new LlamaContext(model, { contextSize: 512, embeddings: true, poolingType: 2 })

  const start2 = Date.now()
  for (const text of texts) {
    ctx.clearMemory()
    const tokens = model.tokenize(text, true)
    ctx.decode(tokens)
    ctx.getEmbeddings(-1)
  }
  const reuseTime = Date.now() - start2
  ctx.free()

  return {
    numTexts,
    newContextTimeMs: newCtxTime,
    newContextRate: numTexts / newCtxTime * 1000,
    reuseContextTimeMs: reuseTime,
    reuseContextRate: numTexts / reuseTime * 1000,
    speedup: newCtxTime / reuseTime,
    perEmbeddingMs: reuseTime / numTexts
  }
}
