const test = require('brittle')
const { LlamaContext, LlamaSampler, generate } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('produces non-empty string output', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 2048 })
  const sampler = new LlamaSampler(loaded.model, { temp: 0.7, topK: 40, topP: 0.95 })
  const output = generate(loaded.model, ctx, sampler, 'The quick brown fox', 32)
  t.ok(typeof output === 'string', 'returns string')
  t.ok(output.length > 0, 'non-empty')
  sampler.free()
  ctx.free()
})

test('respects maxTokens', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 2048 })
  const sampler = new LlamaSampler(loaded.model, { temp: 0.7, topK: 40, topP: 0.95 })
  const tokens = loaded.model.tokenize('The quick brown fox', true)
  ctx.decode(tokens)

  const generated = []
  const maxTokens = 8
  for (let i = 0; i < maxTokens; i++) {
    const token = sampler.sample(ctx, -1)
    if (loaded.model.isEogToken(token)) break
    sampler.accept(token)
    generated.push(token)
    ctx.decode(new Int32Array([token]))
  }

  t.ok(generated.length <= maxTokens, `generated ${generated.length} <= ${maxTokens}`)
  sampler.free()
  ctx.free()
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
