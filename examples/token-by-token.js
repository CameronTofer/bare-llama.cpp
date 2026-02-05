const { LlamaModel, LlamaContext, LlamaSampler } = require('..')

// Path to your GGUF model file - pass after -- (e.g., bare lowlevel.js -- model.gguf)
const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
const modelPath = args[0] || './model.gguf'

console.log('Loading model:', modelPath)
const model = new LlamaModel(modelPath, { nGpuLayers: 99 })

console.log('Creating context...')
const ctx = new LlamaContext(model, { contextSize: 2048 })

console.log('Creating sampler...')
const sampler = new LlamaSampler(model, { temp: 0.8 })

// Tokenize prompt
const prompt = 'Hello, world!'
const tokens = model.tokenize(prompt, true)
console.log('Tokens:', tokens)

// Decode prompt
ctx.decode(tokens)

// Generate tokens one at a time
const generated = []
for (let i = 0; i < 50; i++) {
  const token = sampler.sample(ctx, -1)

  if (model.isEogToken(token)) break

  sampler.accept(token)
  generated.push(token)

  // Decode the new token
  ctx.decode(new Int32Array([token]))
}

// Convert tokens back to text
const text = model.detokenize(new Int32Array(generated))
console.log('Generated:', text)

// Cleanup
sampler.free()
ctx.free()
model.free()
