const test = require('brittle')
const { LlamaContext } = require('..')
const { tryLoadModel } = require('./helpers')

const RERANKER_MODEL = 'qllama/bge-reranker-v2-m3'

const loaded = tryLoadModel(RERANKER_MODEL)

function rerank (model, ctx, query, document) {
  ctx.clearMemory()
  const tokens = model.tokenize(query + '\n' + document, true)
  ctx.decode(tokens)
  return ctx.getEmbeddings(0)
}

test('rerank returns Float32Array', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })
  const score = rerank(loaded.model, ctx, 'What is machine learning?', 'Machine learning is a subset of AI.')
  t.ok(score instanceof Float32Array, 'returns Float32Array')
  ctx.free()
})

test('rerank score is a single float', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })
  const score = rerank(loaded.model, ctx, 'What is machine learning?', 'Machine learning is a subset of AI.')
  t.is(score.length, 1, 'score has length 1')
  t.ok(Number.isFinite(score[0]), 'score is a finite number')
  ctx.free()
})

test('relevant document scores higher than irrelevant', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })

  const query = 'What is machine learning?'
  const relevant = 'Machine learning is a branch of artificial intelligence that enables systems to learn from data.'
  const irrelevant = 'The recipe calls for two cups of flour and one egg.'

  const scoreRelevant = rerank(loaded.model, ctx, query, relevant)[0]
  const scoreIrrelevant = rerank(loaded.model, ctx, query, irrelevant)[0]

  t.ok(scoreRelevant > scoreIrrelevant,
    `relevant (${scoreRelevant.toFixed(4)}) > irrelevant (${scoreIrrelevant.toFixed(4)})`)
  ctx.free()
})

test('reranking is consistent across calls', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })

  const query = 'What is deep learning?'
  const doc = 'Deep learning uses neural networks with many layers.'

  const score1 = rerank(loaded.model, ctx, query, doc)[0]
  const score2 = rerank(loaded.model, ctx, query, doc)[0]

  t.ok(Math.abs(score1 - score2) < 1e-5,
    `scores are consistent: ${score1.toFixed(6)} vs ${score2.toFixed(6)}`)
  ctx.free()
})

test('clearMemory is required for accurate scores', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })

  const query = 'What is machine learning?'
  const doc = 'Machine learning is a subset of AI.'

  // Score with clean context (correct)
  const clean = rerank(loaded.model, ctx, query, doc)[0]

  // Pollute the context with unrelated content, then score WITHOUT clearing
  const junk = loaded.model.tokenize('The weather in Paris is lovely this time of year.', true)
  ctx.decode(junk)
  const dirty = ctx.getEmbeddings(0)[0]

  // Score again with clean context
  const clean2 = rerank(loaded.model, ctx, query, doc)[0]

  t.ok(Math.abs(clean - clean2) < 1e-5,
    `clean scores match: ${clean.toFixed(6)} vs ${clean2.toFixed(6)}`)
  t.ok(Math.abs(clean - dirty) > 1e-3,
    `dirty score (${dirty.toFixed(6)}) differs from clean (${clean.toFixed(6)})`)
  ctx.free()
})

test('reranking produces correct ordering for multiple documents', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512, embeddings: true, poolingType: 4 })

  const query = 'How does photosynthesis work?'
  const docs = [
    'Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen in plant cells.',
    'Plants are green because of chlorophyll, a pigment found in their leaves.',
    'The stock market experienced significant volatility last quarter due to rising interest rates.'
  ]

  const scores = docs.map((doc) => rerank(loaded.model, ctx, query, doc)[0])

  t.ok(scores[0] > scores[2], `best match (${scores[0].toFixed(4)}) > worst match (${scores[2].toFixed(4)})`)
  t.ok(scores[1] > scores[2], `partial match (${scores[1].toFixed(4)}) > worst match (${scores[2].toFixed(4)})`)
  ctx.free()
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
