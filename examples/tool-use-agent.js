const { LlamaModel, LlamaContext, LlamaSampler, generate, setQuiet } = require('..')
const { loadModel, applyTemplate } = require('../lib/ollama.js')

// Define available tools
const tools = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather for a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City name' },
          unit: { type: 'string', enum: ['celsius', 'fahrenheit'], description: 'Temperature unit' }
        },
        required: ['location']
      }
    }
  },
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Perform a mathematical calculation',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Math expression to evaluate' }
        },
        required: ['expression']
      }
    }
  }
]

// Tool implementations
function executeTool (name, args) {
  switch (name) {
    case 'get_weather':
      return JSON.stringify({ temperature: 22, condition: 'sunny', location: args.location })
    case 'calculate':
      try {
        return JSON.stringify({ result: eval(args.expression) })
      } catch (e) {
        return JSON.stringify({ error: e.message })
      }
    default:
      return JSON.stringify({ error: `Unknown tool: ${name}` })
  }
}

// Parse tool calls from model output
function parseToolCalls (output) {
  const calls = []
  const regex = /<tool_call>\s*(\{[\s\S]*?\})\s*<\/tool_call>/g
  let match
  while ((match = regex.exec(output)) !== null) {
    try {
      calls.push(JSON.parse(match[1]))
    } catch {}
  }
  return calls
}

async function agent (args) {
  setQuiet(true)

  const ollama = loadModel(args.model || 'qwen3:0.6b')
  const model = new LlamaModel(ollama.modelPath)
  const context = new LlamaContext(model, { contextSize: 4096 })
  const sampler = new LlamaSampler(model)

  const messages = [{ role: 'user', content: args.prompt }]

  // First pass: let model decide if it needs tools
  let prompt = applyTemplate(ollama, messages, { tools })
  console.log('=== PROMPT ===')
  console.log(prompt)

  let result = generate(model, context, sampler, prompt, 512)
  console.log('=== RESPONSE ===')
  console.log(result)

  // Check for tool calls
  const toolCalls = parseToolCalls(result)
  if (toolCalls.length > 0) {
    console.log('=== TOOL CALLS ===')

    // Add assistant message with tool calls
    messages.push({ role: 'assistant', content: result })

    // Execute tools and add responses
    for (const call of toolCalls) {
      console.log(`Calling ${call.name}(${JSON.stringify(call.arguments)})`)
      const toolResult = executeTool(call.name, call.arguments)
      console.log(`Result: ${toolResult}`)
      messages.push({ role: 'tool', content: toolResult })
    }

    // Second pass: let model respond with tool results
    prompt = applyTemplate(ollama, messages, { tools })
    console.log('=== PROMPT 2 ===')
    console.log(prompt)

    result = generate(model, context, sampler, prompt, 512)
    console.log('=== FINAL RESPONSE ===')
    console.log(result)
  }

  // Cleanup
  sampler.free()
  context.free()
  model.free()

  return result
}

module.exports = agent

// Run if called directly
if (require.main === module) {
  const args = global.Bare.argv.slice(global.Bare.argv.indexOf('--') + 1)
  const prompt = args.join(' ') || 'What is the weather in Tokyo?'
  agent({ prompt }).catch(console.error)
}
