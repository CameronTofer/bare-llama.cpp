// Test models configuration
// Update this list with models you have available locally

module.exports = {
  // Chat/generation model (any instruct-tuned model works)
  // Used for: generation, JSON schema, Lark grammar tests
  generationModel: 'llama3.2:1b',

  // Embedding model
  // Used for: embedding tests, clearMemory benchmark
  embeddingModel: 'nomic-embed-text',

  // Number of iterations for benchmarks
  benchmarkIterations: 50,

  // Max tokens to generate in generation tests
  maxGenerateTokens: 64
}
