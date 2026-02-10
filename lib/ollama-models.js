const fs = require('fs')
const path = require('path')
const os = require('os')

// Get Ollama models directory
function getOllamaDir () {
  const home = os.homedir()
  
  // macOS and Linux default
  const defaultPath = path.join(home, '.ollama', 'models')
  
  if (fs.existsSync(defaultPath)) {
    return defaultPath
  }
  
  // Linux alternative location
  const linuxPath = '/usr/share/ollama/.ollama/models'
  if (fs.existsSync(linuxPath)) {
    return linuxPath
  }
  
  return null
}

// Parse an Ollama model name (e.g., "llama3:latest", "qwen2:7b", "MedAIBase/Qwen3-VL-Reranker:2b")
function parseModelName (name) {
  const parts = name.split(':')
  const fullModel = parts[0]
  const tag = parts[1] || 'latest'

  // Check for namespace (e.g., "MedAIBase/Qwen3-VL-Reranker")
  const slashIdx = fullModel.indexOf('/')
  if (slashIdx !== -1) {
    return {
      namespace: fullModel.slice(0, slashIdx),
      model: fullModel.slice(slashIdx + 1),
      tag
    }
  }

  return {
    namespace: 'library',
    model: fullModel,
    tag
  }
}

// Get manifest for a model
function getManifest (modelName) {
  const ollamaDir = getOllamaDir()
  if (!ollamaDir) return null
  
  const { namespace, model, tag } = parseModelName(modelName)
  
  // Manifest path: manifests/registry.ollama.ai/<namespace>/<model>/<tag>
  const manifestPath = path.join(
    ollamaDir,
    'manifests',
    'registry.ollama.ai',
    namespace,
    model,
    tag
  )
  
  if (!fs.existsSync(manifestPath)) {
    return null
  }
  
  const content = fs.readFileSync(manifestPath, 'utf8')
  return JSON.parse(content)
}

// Get the GGUF model file path from an Ollama model
function getModelPath (modelName) {
  const manifest = getManifest(modelName)
  if (!manifest) {
    throw new Error(`Model "${modelName}" not found in Ollama`)
  }
  
  // Find the model layer (mediaType: application/vnd.ollama.image.model)
  const modelLayer = manifest.layers.find(
    layer => layer.mediaType === 'application/vnd.ollama.image.model'
  )
  
  if (!modelLayer) {
    throw new Error(`No model layer found in "${modelName}" manifest`)
  }
  
  // Convert digest to blob path
  // digest format: "sha256:abc123..."
  // blob path: blobs/sha256-abc123...
  const digest = modelLayer.digest.replace(':', '-')
  const blobPath = path.join(getOllamaDir(), 'blobs', digest)
  
  if (!fs.existsSync(blobPath)) {
    throw new Error(`Model blob not found: ${blobPath}`)
  }
  
  return blobPath
}

// Format a model name for display (omit :latest since it's just noise)
function formatModelName (namespace, model, tag) {
  const prefix = namespace === 'library' ? '' : namespace + '/'
  if (tag === 'latest') return `${prefix}${model}`
  return `${prefix}${model}:${tag}`
}

// List all available Ollama models
function listModels () {
  const ollamaDir = getOllamaDir()
  if (!ollamaDir) return []
  
  const registryDir = path.join(ollamaDir, 'manifests', 'registry.ollama.ai')
  
  if (!fs.existsSync(registryDir)) return []
  
  const models = []
  
  // Iterate through all namespaces (library, user namespaces, etc.)
  const namespaceDirs = fs.readdirSync(registryDir)
  for (const namespace of namespaceDirs) {
    if (namespace.startsWith('.')) continue
    
    const namespacePath = path.join(registryDir, namespace)
    if (!fs.statSync(namespacePath).isDirectory()) continue
    
    // Iterate through model directories within this namespace
    const modelDirs = fs.readdirSync(namespacePath)
    for (const modelName of modelDirs) {
      if (modelName.startsWith('.')) continue
      
      const modelPath = path.join(namespacePath, modelName)
      if (!fs.statSync(modelPath).isDirectory()) continue
      
      // Get tags for this model
      const tags = fs.readdirSync(modelPath)
      for (const tag of tags) {
        if (tag.startsWith('.')) continue
        
        const tagPath = path.join(modelPath, tag)
        if (!fs.statSync(tagPath).isFile()) continue
        
        models.push(formatModelName(namespace, modelName, tag))
      }
    }
  }
  
  return models.sort()
}

// Get model info (size, layers, etc.)
function getModelInfo (modelName) {
  const manifest = getManifest(modelName)
  if (!manifest) return null
  
  const modelLayer = manifest.layers.find(
    layer => layer.mediaType === 'application/vnd.ollama.image.model'
  )
  
  return {
    name: modelName,
    size: modelLayer ? modelLayer.size : 0,
    sizeHuman: modelLayer ? formatBytes(modelLayer.size) : 'unknown',
    layers: manifest.layers.length,
    path: modelLayer ? getModelPath(modelName) : null
  }
}

function formatBytes (bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
}

module.exports = {
  getOllamaDir,
  getManifest,
  getModelPath,
  listModels,
  getModelInfo
}

