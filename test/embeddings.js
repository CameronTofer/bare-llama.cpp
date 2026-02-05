const test = require('brittle')
const { LlamaContext } = require('..')
const { EMBEDDING_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(EMBEDDING_MODEL)

function cosineSimilarity (a, b) {
  let dot = 0, normA = 0, normB = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    normA += a[i] * a[i]
    normB += b[i] * b[i]
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB))
}

function embed (model, ctx, text) {
  ctx.clearMemory()
  const tokens = model.tokenize(text, true)
  ctx.decode(tokens)
  return new Float32Array(ctx.getEmbeddings(-1))
}

test('getEmbeddings returns Float32Array', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 2 })
  const emb = embed(loaded.model, ctx, 'Hello')
  t.ok(emb instanceof Float32Array, 'returns Float32Array')
  ctx.free()
})

test('dimension matches model', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 2 })
  const emb = embed(loaded.model, ctx, 'Hello')
  t.is(emb.length, loaded.model.embeddingDimension, 'dimension matches')
  ctx.free()
})

test('semantic similarity ordering', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 2 })
  const cat = embed(loaded.model, ctx, 'The cat sat on the mat.')
  const feline = embed(loaded.model, ctx, 'A feline rested on the rug.')
  const ml = embed(loaded.model, ctx, 'Machine learning is a subset of artificial intelligence.')

  const simCatFeline = cosineSimilarity(cat, feline)
  const simCatML = cosineSimilarity(cat, ml)
  t.ok(simCatFeline > simCatML, `cat/feline (${simCatFeline.toFixed(4)}) > cat/ML (${simCatML.toFixed(4)})`)
  ctx.free()
})

test('clearMemory reuse produces consistent results', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 2 })
  const emb1 = embed(loaded.model, ctx, 'Hello world')
  const emb2 = embed(loaded.model, ctx, 'Hello world')
  const sim = cosineSimilarity(emb1, emb2)
  t.ok(sim > 0.999, `same text similarity ${sim.toFixed(6)} > 0.999`)
  ctx.free()
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
