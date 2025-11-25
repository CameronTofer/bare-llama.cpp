const { LlamaModel, LlamaContext, LlamaSampler, generate } = require('.')

// Path to your GGUF model file
const modelPath = process.argv[2] || './model.gguf'

console.log('Loading model:', modelPath)
const model = new LlamaModel(modelPath, {
  nGpuLayers: 99  // Use GPU if available
})

console.log('Creating context...')
const ctx = new LlamaContext(model, {
  contextSize: 2048,
  batchSize: 512
})

console.log('Creating sampler...')
const sampler = new LlamaSampler({
  temp: 0.7,
  topK: 40,
  topP: 0.95
})

const prompt = 'The quick brown fox'
console.log('Prompt:', prompt)
console.log('Generating...')

const output = generate(model, ctx, sampler, prompt, 64)
console.log('Output:', output)

// Cleanup
sampler.free()
ctx.free()
model.free()
