const test = require('brittle')
const { readGgufMeta, getModelName } = require('..')
const { GENERATION_MODEL, tryLoadModel } = require('./helpers')

const loaded = tryLoadModel(GENERATION_MODEL)

test('readGgufMeta returns string for known key', { skip: !loaded }, function (t) {
  const name = readGgufMeta(loaded.modelPath, 'general.name')
  t.ok(typeof name === 'string', 'returns string')
  t.ok(name.length > 0, 'non-empty')
})

test('readGgufMeta returns null for unknown key', { skip: !loaded }, function (t) {
  const val = readGgufMeta(loaded.modelPath, 'nonexistent.key.that.does.not.exist')
  t.is(val, null, 'returns null')
})

test('getModelName works', { skip: !loaded }, function (t) {
  const name = getModelName(loaded.modelPath)
  t.ok(typeof name === 'string' || name === null, 'returns string or null')
})

test('cleanup', { skip: !loaded }, function (t) {
  loaded.model.free()
  t.pass('model freed')
})
