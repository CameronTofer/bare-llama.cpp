const { Template } = require('@huggingface/jinja')
const os = require('bare-os')
const fs = require('bare-fs')

// Read GGUF metadata from model file
function readGGUFMetadata (path) {
  const fd = fs.openSync(path, 'r')
  const buf = Buffer.alloc(32 * 1024 * 1024)
  fs.readSync(fd, buf, 0, buf.length, 0)
  fs.closeSync(fd)

  let offset = 0
  const read = (n) => { const v = buf.subarray(offset, offset + n); offset += n; return v }
  const u32 = () => read(4).readUInt32LE(0)
  const u64 = () => Number(read(8).readBigUInt64LE(0))
  const str = () => { const len = u64(); return read(len).toString('utf-8') }

  if (read(4).toString('ascii') !== 'GGUF') throw new Error('Not a GGUF file')
  u32(); u64() // version, tensor count
  const kvCount = u64()

  const metadata = {}
  const readValue = (type) => {
    switch (type) {
      case 0: return read(1).readUInt8(0)
      case 1: return read(1).readInt8(0)
      case 2: return read(2).readUInt16LE(0)
      case 3: return read(2).readInt16LE(0)
      case 4: return read(4).readUInt32LE(0)
      case 5: return read(4).readInt32LE(0)
      case 6: return read(4).readFloatLE(0)
      case 7: return read(1).readUInt8(0) !== 0
      case 8: return str()
      case 9: { const t = u32(), n = u64(), a = []; for (let i = 0; i < n; i++) a.push(readValue(t)); return a }
      case 10: return u64()
      case 11: return Number(read(8).readBigInt64LE(0))
      case 12: return read(8).readDoubleLE(0)
      default: return null
    }
  }

  for (let i = 0; i < kvCount; i++) {
    metadata[str()] = readValue(u32())
  }
  return metadata
}

// Load model from Ollama's local storage
function loadModel (name) {
  const homedir = os.homedir()
  const ollamaPath = `${homedir}/.ollama/models`
  const manifestsPath = `${ollamaPath}/manifests/registry.ollama.ai/library`

  const [family, tag] = name.includes(':') ? name.split(':') : [name, 'latest']
  let manifest
  try {
    manifest = JSON.parse(fs.readFileSync(`${manifestsPath}/${family}/${tag}`, 'utf-8'))
  } catch {
    throw new Error(`Model ${name} not found. Run 'ollama pull ${name}' first.`)
  }

  const blobPath = (d) => `${ollamaPath}/blobs/${d.replace('sha256:', 'sha256-')}`
  const modelLayer = manifest.layers.find(l => l.mediaType.endsWith('.model'))
  if (!modelLayer) throw new Error(`No model layer in ${name}`)

  const modelPath = blobPath(modelLayer.digest)
  const metadata = readGGUFMetadata(modelPath)
  const tokens = metadata['tokenizer.ggml.tokens'] || []

  return {
    name,
    family,
    tag,
    modelPath,
    chatTemplate: metadata['tokenizer.chat_template'],
    bosToken: tokens[metadata['tokenizer.ggml.bos_token_id']] || '',
    eosToken: tokens[metadata['tokenizer.ggml.eos_token_id']] || ''
  }
}

// Simplified Qwen3/ChatML template that works with @huggingface/jinja
const SIMPLE_CHATML_TEMPLATE = `
{%- if tools %}
<|im_start|>system
{%- if messages[0].role == 'system' %}
{{ messages[0].content }}

{% endif %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{ tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
{%- else %}
{%- if messages[0].role == 'system' %}
<|im_start|>system
{{ messages[0].content }}<|im_end|>
{%- endif %}
{%- endif %}
{%- for message in messages %}
{%- if message.role == 'user' %}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == 'assistant' %}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- elif message.role == 'tool' %}
<|im_start|>user
<tool_response>
{{ message.content }}
</tool_response><|im_end|>
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
<|im_start|>assistant
{%- endif %}
`.trim()

// Apply Jinja chat template to messages
function applyTemplate (model, messages, options = {}) {
  const context = {
    messages,
    tools: options.tools || null,
    bos_token: model.bosToken,
    eos_token: model.eosToken,
    add_generation_prompt: true
  }

  // Try the GGUF template first if available
  if (model.chatTemplate) {
    try {
      return new Template(model.chatTemplate).render(context)
    } catch (e) {
      // Template uses unsupported Python methods, fall back to simple template
      console.warn(`GGUF template failed (${e.message}), using fallback`)
    }
  }

  // Fall back to simplified ChatML template
  return new Template(SIMPLE_CHATML_TEMPLATE).render(context)
}

module.exports = {
  readGGUFMetadata,
  loadModel,
  applyTemplate
}
