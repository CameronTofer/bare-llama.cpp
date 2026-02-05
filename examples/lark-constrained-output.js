const { LlamaModel, LlamaContext, LlamaSampler, generate, setQuiet } = require('..')

setQuiet(true)

const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
const modelPath = args[0] || './model.gguf'

console.log('Loading model:', modelPath)
const model = new LlamaModel(modelPath, { nGpuLayers: 99 })

console.log('Creating context...')
const ctx = new LlamaContext(model, { contextSize: 2048 })

// Lark grammar using llguidance
// This grammar constrains output to "yes" or "no" only
// See: https://github.com/guidance-ai/llguidance/blob/main/docs/syntax.md
const grammar = `
start: RESPONSE
RESPONSE: "yes" | "no"
`

console.log('Creating sampler with Lark grammar constraint...')
console.log('Grammar:', grammar.trim())

const sampler = new LlamaSampler(model, {
  temp: 0,  // greedy for deterministic output
  lark: grammar
})

const prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nIs the sky blue? Answer yes or no only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
console.log('Generating constrained response...')

const output = generate(model, ctx, sampler, prompt, 4)
console.log('Output:', output.trim())

// Cleanup
sampler.free()
ctx.free()
model.free()
