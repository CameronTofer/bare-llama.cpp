const { LlamaModel, LlamaContext, LlamaSampler, setQuiet } = require('..')
const { getModelPath } = require('./ollama-models')

setQuiet(true)

console.log('# Double-Free Crash Test\n')
console.log('This test attempts to reliably trigger the double-free bug.\n')

const modelPath = getModelPath('llama3.2:1b')

// Create a function that will go out of scope, forcing GC
function testScope() {
  console.log('Loading model inside function scope...')
  const model = new LlamaModel(modelPath, { nGpuLayers: 0 })
  console.log('✓ Model loaded')
  
  console.log('Calling model.free()...')
  model.free()
  console.log('✓ Manual free completed')
  
  console.log('Returning from function (model goes out of scope)...')
  // Model object goes out of scope here, making it eligible for GC
}

console.log('Test 1: Model going out of scope after manual free')
testScope()
console.log('✓ Function returned\n')

// Create many objects to trigger GC
console.log('Creating memory pressure to trigger GC...')
const garbage = []
for (let i = 0; i < 1000; i++) {
  garbage.push(new Array(10000).fill(Math.random()))
}
console.log('✓ Memory pressure created\n')

// Clear garbage
garbage.length = 0
console.log('✓ Garbage cleared\n')

// Try multiple iterations
console.log('Running multiple iterations to increase chance of GC...')
for (let i = 0; i < 5; i++) {
  console.log(`  Iteration ${i + 1}...`)
  testScope()
  // Create and clear garbage each iteration
  const temp = []
  for (let j = 0; j < 100; j++) {
    temp.push(new Array(10000).fill(j))
  }
  temp.length = 0
}
console.log('✓ All iterations completed\n')

console.log('If this script exits with code 0, the bug was not triggered.')
console.log('If it crashes with SIGABRT (exit code 134), the double-free bug was triggered.')
console.log('The bug exists regardless - it depends on GC timing.\n')
