const { command, flag, arg, summary, description } = require('paparam')
const Corestore = require('corestore')
const Hyperdrive = require('hyperdrive')
const Hyperswarm = require('hyperswarm')
const Signals = require('bare-signals')
const fs = require('bare-fs')
const path = require('bare-path')
const { getOllamaDir, getManifest, listModels, getModelInfo } = require('../lib/ollama-models.js')

// Map mediaType to filename
const LAYER_FILENAMES = {
  'application/vnd.ollama.image.model': 'model.gguf',
  'application/vnd.ollama.image.template': 'template.txt',
  'application/vnd.ollama.image.params': 'params.json',
  'application/vnd.ollama.image.license': 'license.txt',
  'application/vnd.ollama.image.system': 'system.txt',
  'application/vnd.ollama.image.projector': 'projector.gguf',
  'application/vnd.ollama.image.adapter': 'adapter.gguf'
}

// Parse an Ollama model name (e.g., "llama3:latest", "qwen2:7b")
function parseModelName (name) {
  const parts = name.split(':')
  return {
    model: parts[0],
    tag: parts[1] || 'latest'
  }
}

function formatBytes (bytes) {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
}

// Get blob path for a layer digest
function getBlobPath (ollamaDir, digest) {
  // digest format: "sha256:abc123..."
  // blob path: blobs/sha256-abc123...
  const blobName = digest.replace(':', '-')
  return path.join(ollamaDir, 'blobs', blobName)
}

// Load or create the drives manifest (tracks all imported models)
function loadDrivesManifest (storagePath) {
  const manifestPath = path.join(storagePath, 'drives.json')
  if (fs.existsSync(manifestPath)) {
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'))
  }
  return { drives: {} }
}

function saveDrivesManifest (storagePath, manifest) {
  const manifestPath = path.join(storagePath, 'drives.json')
  fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2))
}

// Import a model from Ollama into a Hyperdrive
async function importModelToDrive (drive, modelName, ollamaDir) {
  const manifest = getManifest(modelName)
  if (!manifest) {
    throw new Error(`Model "${modelName}" not found in Ollama`)
  }

  const { model, tag } = parseModelName(modelName)

  // Track what we import
  const imported = []
  let totalSize = 0

  // Import all layers
  for (const layer of manifest.layers) {
    const blobPath = getBlobPath(ollamaDir, layer.digest)

    if (!fs.existsSync(blobPath)) {
      console.warn(`Blob not found for ${layer.mediaType}: ${blobPath}`)
      continue
    }

    // Determine filename from mediaType
    let filename = LAYER_FILENAMES[layer.mediaType]
    if (!filename) {
      // Unknown type - use digest as filename
      const digestShort = layer.digest.split(':')[1].slice(0, 12)
      filename = `unknown-${digestShort}.bin`
    }

    console.log(`  Importing ${filename} (${formatBytes(layer.size)})...`)

    // Stream file into drive
    const readStream = fs.createReadStream(blobPath)
    const writeStream = drive.createWriteStream(`/${filename}`)

    await new Promise((resolve, reject) => {
      readStream.pipe(writeStream)
      writeStream.on('close', resolve)
      writeStream.on('error', reject)
      readStream.on('error', reject)
    })

    imported.push({
      filename,
      mediaType: layer.mediaType,
      size: layer.size,
      digest: layer.digest
    })
    totalSize += layer.size
  }

  // Write metadata
  const metadata = {
    name: modelName,
    model,
    tag,
    totalSize,
    totalSizeHuman: formatBytes(totalSize),
    layers: imported,
    manifest,
    importedAt: new Date().toISOString()
  }

  await drive.put('/model.json', Buffer.from(JSON.stringify(metadata, null, 2)))

  return metadata
}

// Seed all drives in the corestore
async function seedAllDrives (store, drivesManifest, swarm, openDrives = new Map()) {
  const drives = []

  for (const [modelName, driveInfo] of Object.entries(drivesManifest.drives)) {
    // Reuse already-open drive if available
    let drive = openDrives.get(modelName)
    if (!drive) {
      const key = Buffer.from(driveInfo.key, 'hex')
      drive = new Hyperdrive(store, key)
      await drive.ready()
    }

    // Join swarm on this drive's discovery key
    const discovery = swarm.join(drive.discoveryKey, { server: true, client: false })
    await discovery.flushed()

    console.log(`  ${modelName}: ${driveInfo.key.slice(0, 16)}...`)
    drives.push(drive)
  }

  return drives
}

const main = command(
  'ollama-hyperdrive',
  summary('Import Ollama models into Hyperdrives and seed them'),
  description('Import Ollama models into separate Hyperdrives sharing a Corestore, then seed all drives on the DHT'),
  flag('--storage <path>', 'Corestore storage path').default('./.hyperdrive'),
  flag('--ollama-dir <path>', 'Ollama models directory'),
  flag('--list', 'List available Ollama models'),
  flag('--status', 'Show status of imported drives'),
  arg('[model]', 'Ollama model name to import (e.g., llama3:latest)'),
  async (cmd) => {
    const storagePath = path.resolve(cmd.flags.storage)
    const ollamaDir = cmd.flags.ollamaDir || getOllamaDir()

    // List available models
    if (cmd.flags.list) {
      console.log('Available Ollama models:')
      const models = listModels()
      if (models.length === 0) {
        console.log('  No models found. Is Ollama installed?')
        return
      }
      for (const model of models) {
        const info = getModelInfo(model)
        console.log(`  ${model} (${info.sizeHuman})`)
      }
      return
    }

    // Ensure storage directory exists
    if (!fs.existsSync(storagePath)) {
      fs.mkdirSync(storagePath, { recursive: true })
    }

    // Load drives manifest
    const drivesManifest = loadDrivesManifest(storagePath)

    // Show status
    if (cmd.flags.status) {
      console.log('Imported drives:')
      const entries = Object.entries(drivesManifest.drives)
      if (entries.length === 0) {
        console.log('  No models imported yet.')
        return
      }
      for (const [modelName, info] of entries) {
        console.log(`  ${modelName}`)
        console.log(`    Key: ${info.key}`)
        console.log(`    Size: ${info.sizeHuman}`)
        console.log(`    Imported: ${info.importedAt}`)
      }
      return
    }

    // Create corestore
    const store = new Corestore(storagePath)
    await store.ready()

    // Create swarm for seeding
    const swarm = new Hyperswarm()

    // Replicate all drives on connection
    swarm.on('connection', (socket, info) => {
      console.log('Peer connected:', info.publicKey.toString('hex').slice(0, 8) + '...')
      store.replicate(socket)
    })

    // Track open drives (keyed by model name)
    const openDrives = new Map()

    // If a model is specified, import it
    const modelName = cmd.args.model
    if (modelName) {
      // Check if already imported
      if (drivesManifest.drives[modelName]) {
        console.log(`Model "${modelName}" already imported.`)
        console.log(`Key: ${drivesManifest.drives[modelName].key}`)
      } else {
        // Validate model exists
        if (!ollamaDir) {
          console.error('Ollama models directory not found.')
          return
        }

        const manifest = getManifest(modelName)
        if (!manifest) {
          console.error(`Model "${modelName}" not found in Ollama.`)
          console.log('Use --list to see available models.')
          return
        }

        console.log(`Importing model: ${modelName}`)

        // Create a new drive for this model using a namespaced store
        // This ensures each model gets its own unique keypair
        const driveStore = store.namespace(modelName)
        const drive = new Hyperdrive(driveStore)
        await drive.ready()

        // Import the model
        const metadata = await importModelToDrive(drive, modelName, ollamaDir)

        // Save to manifest
        drivesManifest.drives[modelName] = {
          key: drive.key.toString('hex'),
          discoveryKey: drive.discoveryKey.toString('hex'),
          sizeHuman: metadata.totalSizeHuman,
          importedAt: metadata.importedAt
        }
        saveDrivesManifest(storagePath, drivesManifest)

        // Keep track of this drive so we don't reopen it
        openDrives.set(modelName, drive)

        console.log('')
        console.log('Import complete!')
        console.log(`Key: ${drive.key.toString('hex')}`)
        console.log(`Size: ${metadata.totalSizeHuman}`)
      }
    }

    // Seed all drives
    const driveEntries = Object.entries(drivesManifest.drives)
    if (driveEntries.length === 0) {
      console.log('No models to seed. Import a model first.')
      console.log('Usage: ollama-hyperdrive <model>')
      console.log('       ollama-hyperdrive --list')
      await store.close()
      return
    }

    console.log('')
    console.log('Seeding all imported models...')

    const drives = await seedAllDrives(store, drivesManifest, swarm, openDrives)

    // Wait for initial DHT announce
    await swarm.flush()

    console.log('')
    console.log('Ready! Seeding the following models:')
    for (const [modelName, info] of driveEntries) {
      console.log(`  ${modelName}: ${info.key}`)
    }
    console.log('')
    console.log('Press Ctrl+C to stop seeding.')

    // Handle graceful shutdown
    const signals = new Signals('SIGINT')
    signals.once('signal', async () => {
      console.log('\nShutting down...')
      await swarm.destroy()
      for (const drive of drives) {
        await drive.close()
      }
      await store.close()
      signals.close()
    })
  }
)

main.parse()
