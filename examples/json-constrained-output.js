const { LlamaModel, LlamaContext, LlamaSampler, generate, setQuiet } = require('..')

setQuiet(true)

const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
const modelPath = args[0] || './model.gguf'

console.log('Loading model:', modelPath)
const model = new LlamaModel(modelPath, { nGpuLayers: 99 })

console.log('Creating context...')
const ctx = new LlamaContext(model, { contextSize: 2048 })

// JSON schema constraint using llguidance
// This constrains the model to output valid JSON matching the schema
const schema = JSON.stringify({
  type: 'object',
  properties: {
    name: { type: 'string' },
    age: { type: 'integer' }
  },
  required: ['name', 'age'],
  additionalProperties: false
})

console.log('Creating sampler with JSON schema constraint...')
console.log('Schema:', schema)

const sampler = new LlamaSampler(model, {
  temp: 0,  // greedy for deterministic output
  json: schema
})

const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nGenerate JSON for a person named Alice who is 30.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
console.log('Generating JSON...')

const output = generate(model, ctx, sampler, prompt, 64)
console.log('Output:', output.trim())

// Cleanup
sampler.free()
ctx.free()
model.free()
