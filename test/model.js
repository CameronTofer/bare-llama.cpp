const test = require('brittle')
const { LlamaModel } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('tokenize returns Int32Array', { skip: !loaded }, function (t) {
  const tokens = loaded.model.tokenize('Hello', true)
  t.ok(tokens instanceof Int32Array, 'returns Int32Array')
  t.ok(tokens.length > 0, 'non-empty')
})

test('tokenize/detokenize roundtrip', { skip: !loaded }, function (t) {
  const cases = [
    'Hello, world!',
    'The quick brown fox jumps over the lazy dog.',
    '日本語テスト',
    'emoji: 🎉🚀',
    ''
  ]
  for (const text of cases) {
    const tokens = loaded.model.tokenize(text, false)
    const decoded = loaded.model.detokenize(tokens)
    t.is(decoded, text, `roundtrip: "${text}"`)
  }
})

test('embeddingDimension is a number', { skip: !loaded }, function (t) {
  const dim = loaded.model.embeddingDimension
  t.ok(typeof dim === 'number', 'is a number')
  t.ok(dim > 0, 'positive')
})

test('trainingContextSize is a number', { skip: !loaded }, function (t) {
  const size = loaded.model.trainingContextSize
  t.ok(typeof size === 'number', 'is a number')
  t.ok(size > 0, 'positive')
})

test('getMeta returns string for known key', { skip: !loaded }, function (t) {
  const name = loaded.model.getMeta('general.name')
  t.ok(typeof name === 'string' || name === null, 'returns string or null')
})

test('name getter works', { skip: !loaded }, function (t) {
  const name = loaded.model.name
  t.ok(typeof name === 'string' || name === null, 'returns string or null')
})

test('isEogToken returns boolean', { skip: !loaded }, function (t) {
  t.ok(typeof loaded.model.isEogToken(0) === 'boolean', 'returns boolean')
})

test('constructor throws on bad path', function (t) {
  t.exception(() => new LlamaModel('/nonexistent/model.gguf'), 'throws on bad path')
})

test('free() is idempotent', { skip: !loaded }, function (t) {
  const { model } = tryLoadModel(GENERATION_MODEL, { nGpuLayers: 0 })
  model.free()
  model.free()
  t.pass('double free did not crash')
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
