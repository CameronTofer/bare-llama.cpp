const { LlamaContext, LlamaSampler } = require('..')

module.exports = function runGenerationBench (model) {
  const ctx = new LlamaContext(model, { contextSize: 2048, batchSize: 512 })
  const sampler = new LlamaSampler(model, { temp: 0.7, topK: 40, topP: 0.95 })

  const prompt = 'The quick brown fox'
  const tokens = model.tokenize(prompt, true)
  const maxTokens = 128

  // Prompt processing
  const decodeStart = Date.now()
  ctx.decode(tokens)
  const promptTime = Date.now() - decodeStart

  // Generation
  const genStart = Date.now()
  const generated = []
  let firstTokenTime = null

  for (let i = 0; i < maxTokens; i++) {
    const token = sampler.sample(ctx, -1)

    if (firstTokenTime === null) {
      firstTokenTime = Date.now() - genStart
    }

    if (model.isEogToken(token)) break

    sampler.accept(token)
    generated.push(token)
    ctx.decode(new Int32Array([token]))
  }

  const genTime = Date.now() - genStart

  sampler.free()
  ctx.free()

  return {
    promptTokens: tokens.length,
    promptTimeMs: promptTime,
    promptSpeed: tokens.length / promptTime * 1000,
    generatedTokens: generated.length,
    genTimeMs: genTime,
    genSpeed: generated.length / genTime * 1000,
    firstTokenMs: firstTokenTime
  }
}
