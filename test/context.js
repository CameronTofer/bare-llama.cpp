const test = require('brittle')
const { LlamaContext } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('constructor requires LlamaModel', function (t) {
  t.exception(() => new LlamaContext({}), 'throws on non-model')
})

test('contextSize getter', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  t.ok(typeof ctx.contextSize === 'number', 'is a number')
  t.ok(ctx.contextSize > 0, 'positive')
  ctx.free()
})

test('decode accepts Int32Array', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  const tokens = loaded.model.tokenize('Hello', true)
  ctx.decode(tokens)
  t.pass('decode did not throw')
  ctx.free()
})

test('clear() resets context for fresh prompt', { skip: !loaded }, function (t) {
  const { LlamaSampler } = require('..')
  const ctx = new LlamaContext(loaded.model, { contextSize: 2048 })
  const sampler = new LlamaSampler(loaded.model, { temp: 0 })

  // Decode a prompt and sample a token
  const tokens1 = loaded.model.tokenize('The capital of France is', true)
  ctx.decode(tokens1)
  const token1 = sampler.sample(ctx, -1)

  // Clear and decode the same prompt again
  ctx.clear()
  const tokens2 = loaded.model.tokenize('The capital of France is', true)
  ctx.decode(tokens2)
  const token2 = sampler.sample(ctx, -1)

  t.is(token1, token2, 'same prompt after clear() produces same token')

  sampler.free()
  ctx.free()
})

test('free() is idempotent', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  ctx.free()
  ctx.free()
  t.pass('double free did not crash')
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
