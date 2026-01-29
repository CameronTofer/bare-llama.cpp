const fs = require('bare-fs')
const path = require('bare-path')
const os = require('bare-os')

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

// Parse an Ollama model name (e.g., "llama3:latest", "qwen2:7b")
function parseModelName (name) {
  const parts = name.split(':')
  return {
    model: parts[0],
    tag: parts[1] || 'latest'
  }
}

// Get manifest for a model
function getManifest (modelName) {
  const ollamaDir = getOllamaDir()
  if (!ollamaDir) return null
  
  const { model, tag } = parseModelName(modelName)
  
  // Manifest path: manifests/registry.ollama.ai/library/<model>/<tag>
  const manifestPath = path.join(
    ollamaDir,
    'manifests',
    'registry.ollama.ai',
    'library',
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

// List all available Ollama models
function listModels () {
  const ollamaDir = getOllamaDir()
  if (!ollamaDir) return []
  
  const manifestsDir = path.join(ollamaDir, 'manifests', 'registry.ollama.ai', 'library')
  
  if (!fs.existsSync(manifestsDir)) return []
  
  const models = []
  
  // Iterate through model directories
  const modelDirs = fs.readdirSync(manifestsDir)
  for (const modelName of modelDirs) {
    if (modelName.startsWith('.')) continue
    
    const modelPath = path.join(manifestsDir, modelName)
    const stat = fs.statSync(modelPath)
    
    if (!stat.isDirectory()) continue
    
    // Get tags for this model
    const tags = fs.readdirSync(modelPath)
    for (const tag of tags) {
      if (tag.startsWith('.')) continue
      
      const tagPath = path.join(modelPath, tag)
      const tagStat = fs.statSync(tagPath)
      
      if (tagStat.isFile()) {
        models.push(`${modelName}:${tag}`)
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

