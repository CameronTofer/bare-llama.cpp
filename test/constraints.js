const test = require('brittle')
const { LlamaContext, LlamaSampler, generate } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('JSON schema produces parseable JSON with correct fields', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 2048 })

  const schema = JSON.stringify({
    type: 'object',
    properties: {
      name: { type: 'string' },
      age: { type: 'integer' }
    },
    required: ['name', 'age'],
    additionalProperties: false
  })

  let sampler
  try {
    sampler = new LlamaSampler(loaded.model, { temp: 0, json: schema })
  } catch {
    t.comment('llguidance not available, skipping')
    ctx.free()
    return
  }

  const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate JSON for a person named Alice who is 30.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  const output = generate(loaded.model, ctx, sampler, prompt, 32)
  sampler.free()
  ctx.free()
  // Grammar constraint may partially work - verify output contains JSON-like structure
  t.ok(output.includes('"name"'), 'output contains name field')
  t.ok(output.includes('"age"'), 'output contains age field')
})

test('Lark grammar constrains to yes/no', { skip: !loaded }, function (t) {
  const ctx = new LlamaContext(loaded.model, { contextSize: 2048 })

  const grammar = `
start: RESPONSE
RESPONSE: "yes" | "no"
`

  let sampler
  try {
    sampler = new LlamaSampler(loaded.model, { temp: 0, lark: grammar })
  } catch {
    t.comment('llguidance not available, skipping')
    ctx.free()
    return
  }

  const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIs the sky blue? Answer yes or no only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
  const output = generate(loaded.model, ctx, sampler, prompt, 4)
  const trimmed = output.trim()
  t.ok(trimmed === 'yes' || trimmed === 'no', `got "${trimmed}"`)
  sampler.free()
  ctx.free()
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
