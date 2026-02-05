const test = require('brittle')
const { LlamaContext, LlamaSampler } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL, { nGpuLayers: 0 })

test('model.free() twice does not crash', { skip: !loaded }, function (t) {
  const { model } = tryLoadModel(GENERATION_MODEL, { nGpuLayers: 0 })
  model.free()
  model.free()
  t.pass('no crash')
})

test('context.free() twice does not crash', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 512 })
  ctx.free()
  ctx.free()
  t.pass('no crash')
})

test('sampler.free() twice does not crash', { skip: !loaded }, function (t) {
  const sampler = new LlamaSampler(loaded.model, { temp: 0 })
  sampler.free()
  sampler.free()
  t.pass('no crash')
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
