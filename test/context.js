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

test('clearMemory does not throw', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  const tokens = loaded.model.tokenize('Hello', true)
  ctx.decode(tokens)
  ctx.clearMemory()
  t.pass('clearMemory did not throw')
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
