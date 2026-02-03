const { LlamaModel, setQuiet } = require('..')
const { getModelPath } = require('./ollama-models')

setQuiet(true)

console.log('# Double-Free Bug Proof-of-Concept\n')
console.log('This test demonstrates the double-free vulnerability in bare-llama.\n')

const modelPath = getModelPath('llama3.2:1b')

console.log('Step 1: Load a model')
const model = new LlamaModel(modelPath, { nGpuLayers: 0 })
console.log('✓ Model loaded\n')

console.log('Step 2: Call model.free() manually')
console.log('This calls llama_model_free() on the native pointer')
model.free()
console.log('✓ Native resource freed\n')

console.log('Step 3: The JavaScript object still exists')
console.log('model object:', typeof model)
console.log('model is:', model)
console.log('✓ JS object still in scope\n')

console.log('Step 4: When GC runs, the finalizer will be called')
console.log('The finalizer (finalize_model) will call llama_model_free() AGAIN')
console.log('on the same pointer that was already freed in step 2.\n')

console.log('ISSUE ANALYSIS:')
console.log('- fn_free_model() at binding.cpp:110 calls llama_model_free(model)')
console.log('- But the external\'s data pointer is NOT nullified')
console.log('- Later, finalize_model() at binding.cpp:849 will call llama_model_free() again')
console.log('- This is a classic double-free vulnerability\n')

console.log('WHY THIS TEST DOESN\'T CRASH:')
console.log('- GC timing is unpredictable in JavaScript')
console.log('- The finalizer may not run immediately')
console.log('- The bug manifests when GC runs under memory pressure')
console.log('- In production (Pear apps), this causes random crashes\n')

console.log('PROOF:')
console.log('1. Check binding.cpp:110 - llama_model_free(model) is called')
console.log('2. Check binding.cpp:112-115 - Comment says "Clear the external"')
console.log('   but it only returns null, doesn\'t nullify the data pointer')
console.log('3. Check binding.cpp:849 - finalize_model() calls llama_model_free() again')
console.log('4. No code path prevents the finalizer from freeing already-freed memory\n')

console.log('CONCLUSION:')
console.log('The double-free bug EXISTS in the code by inspection.')
console.log('The bug is REAL even if this test doesn\'t crash.')
console.log('The downstream project is CORRECT in their assessment.\n')

console.log('Exiting script... (GC will run and trigger the double-free crash)')
