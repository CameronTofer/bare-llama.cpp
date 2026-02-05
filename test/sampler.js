const test = require('brittle')
const { LlamaContext, LlamaSampler } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('constructor requires LlamaModel', function (t) {
  t.exception(() => new LlamaSampler({}), 'throws on non-model')
})

test('sample returns number', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  const sampler = new LlamaSampler(loaded.model, { temp: 0 })
  const tokens = loaded.model.tokenize('Hello', true)
  ctx.decode(tokens)
  const token = sampler.sample(ctx, -1)
  t.ok(typeof token === 'number', 'returns number')
  sampler.free()
  ctx.free()
})

test('accept does not throw', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  const sampler = new LlamaSampler(loaded.model, { temp: 0 })
  const tokens = loaded.model.tokenize('Hello', true)
  ctx.decode(tokens)
  const token = sampler.sample(ctx, -1)
  sampler.accept(token)
  t.pass('accept did not throw')
  sampler.free()
  ctx.free()
})

test('free() is idempotent', { skip: !loaded }, function (t) {
  const sampler = new LlamaSampler(loaded.model, { temp: 0 })
  sampler.free()
  sampler.free()
  t.pass('double free did not crash')
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
