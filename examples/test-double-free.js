const { LlamaModel, LlamaContext, LlamaSampler, setQuiet } = require('..')
const { getModelPath } = require('./ollama-models')

setQuiet(true)

console.log('# Double-Free Test\n')
console.log('Testing if manual .free() followed by GC causes a crash...\n')

const modelPath = getModelPath('llama3.2:1b')

// Test 1: Model double-free
console.log('Test 1: Model double-free')
console.log('Creating and freeing models in a loop to trigger GC...')
const freedModels = []
for (let i = 0; i < 3; i++) {
  console.log(`  Iteration ${i + 1}: Loading model...`)
  const model = new LlamaModel(modelPath, { nGpuLayers: 0 })
  console.log(`  Iteration ${i + 1}: Calling model.free()...`)
  model.free()
  freedModels.push(model) // Keep reference to freed model
  console.log(`  Iteration ${i + 1}: ✓ Manual free completed`)
}

// Test 2: Context double-free
console.log('\nTest 2: Context double-free')
console.log('Loading model for context test...')
const model = new LlamaModel(modelPath, { nGpuLayers: 0 })
console.log('Creating and freeing contexts in a loop...')
const freedContexts = []
for (let i = 0; i < 3; i++) {
  console.log(`  Iteration ${i + 1}: Creating context...`)
  const ctx = new LlamaContext(model, { contextSize: 512 })
  console.log(`  Iteration ${i + 1}: Calling ctx.free()...`)
  ctx.free()
  freedContexts.push(ctx) // Keep reference to freed context
  console.log(`  Iteration ${i + 1}: ✓ Manual free completed`)
}

// Test 3: Sampler double-free
console.log('\nTest 3: Sampler double-free')
console.log('Creating and freeing samplers in a loop...')
const freedSamplers = []
for (let i = 0; i < 3; i++) {
  console.log(`  Iteration ${i + 1}: Creating sampler...`)
  const sampler = new LlamaSampler(model, { temp: 0.7 })
  console.log(`  Iteration ${i + 1}: Calling sampler.free()...`)
  sampler.free()
  freedSamplers.push(sampler) // Keep reference to freed sampler
  console.log(`  Iteration ${i + 1}: ✓ Manual free completed`)
}

model.free()

// Try to trigger GC
console.log('\nAttempting to trigger GC...')
if (global.gc) {
  global.gc()
  console.log('✓ GC triggered explicitly')
} else {
  console.log('⚠ GC not available (run with --expose-gc)')
  console.log('Creating memory pressure to trigger GC...')
  // Create memory pressure
  const arrays = []
  for (let i = 0; i < 100; i++) {
    arrays.push(new Array(100000).fill(i))
  }
  arrays.length = 0 // Clear references
}

// Wait for potential crash
console.log('\nWaiting for potential crash...')
setTimeout(() => {
  console.log('✓ No crash detected after 500ms')
  console.log('\n⚠ WARNING: This test may not reliably trigger the double-free bug')
  console.log('The bug depends on GC timing and may only manifest in production')
  console.log('under memory pressure or when objects go out of scope naturally.')
  process.exit(0)
}, 500)
