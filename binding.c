#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <bare.h>
#include <js.h>
#include <utf.h>
#include <llama.h>

// Custom type tags for prevent type confusion
static js_type_tag_t llama_model_type_tag = {0x4c4c414d41, 0x4d4f44454c};  // "LLAMA MODEL"
static js_type_tag_t llama_context_type_tag = {0x4c4c414d41, 0x435458};    // "LLAMA CTX"
static js_type_tag_t llama_sampler_type_tag = {0x4c4c414d41, 0x53414d50};  // "LLAMA SAMP"

// Forward declarations
static void finalize_model(js_env_t *env, void *data, void *hint);
static void finalize_context(js_env_t *env, void *data, void *hint);
static void finalize_sampler(js_env_t *env, void *data, void *hint);

// Helper to throw JS error
static js_value_t *throw_error(js_env_t *env, const char *msg) {
  js_throw_error(env, NULL, msg);
  return NULL;
}

// loadModel(path: string, params?: object): Model
static js_value_t *
fn_load_model(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 1) return throw_error(env, "Model path required");

  // Get model path
  size_t path_len;
  err = js_get_value_string_utf8(env, argv[0], NULL, 0, &path_len);
  if (err < 0) return throw_error(env, "Invalid model path");

  char *path = malloc(path_len + 1);
  if (!path) return throw_error(env, "Memory allocation failed");

  err = js_get_value_string_utf8(env, argv[0], (utf8_t *)path, path_len + 1, NULL);
  if (err < 0) {
    free(path);
    return throw_error(env, "Failed to read model path");
  }

  // Set up default params
  struct llama_model_params params = llama_model_default_params();

  // Parse optional params
  if (argc >= 2) {
    js_value_t *opts = argv[1];
    js_value_t *val;
    bool has_prop;

    // n_gpu_layers
    err = js_has_named_property(env, opts, "nGpuLayers", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "nGpuLayers", &val);
      if (err == 0) {
        int32_t n;
        js_get_value_int32(env, val, &n);
        params.n_gpu_layers = n;
      }
    }
  }

  // Load the model
  struct llama_model *model = llama_model_load_from_file(path, params);
  free(path);

  if (!model) return throw_error(env, "Failed to load model");

  // Wrap in JS object
  js_value_t *result;
  err = js_create_external(env, model, finalize_model, NULL, &result);
  if (err < 0) {
    llama_model_free(model);
    return throw_error(env, "Failed to create model wrapper");
  }

  err = js_add_type_tag(env, result, &llama_model_type_tag);
  assert(err == 0);

  return result;
}

// freeModel(model: Model): void
static js_value_t *
fn_free_model(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return NULL;

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_model_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid model");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return NULL;

  llama_model_free(model);

  // Clear the external to prevent double-free
  js_value_t *null_val;
  js_get_null(env, &null_val);
  return null_val;
}

// createContext(model: Model, params?: object): Context
static js_value_t *
fn_create_context(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 1) return throw_error(env, "Model required");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_model_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid model");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  struct llama_context_params params = llama_context_default_params();

  // Parse optional params
  if (argc >= 2) {
    js_value_t *opts = argv[1];
    js_value_t *val;
    bool has_prop;

    // n_ctx (context size)
    err = js_has_named_property(env, opts, "contextSize", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "contextSize", &val);
      if (err == 0) {
        int32_t n;
        js_get_value_int32(env, val, &n);
        params.n_ctx = (uint32_t)n;
      }
    }

    // n_batch
    err = js_has_named_property(env, opts, "batchSize", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "batchSize", &val);
      if (err == 0) {
        int32_t n;
        js_get_value_int32(env, val, &n);
        params.n_batch = (uint32_t)n;
      }
    }
  }

  struct llama_context *ctx = llama_init_from_model(model, params);
  if (!ctx) return throw_error(env, "Failed to create context");

  js_value_t *result;
  err = js_create_external(env, ctx, finalize_context, NULL, &result);
  if (err < 0) {
    llama_free(ctx);
    return throw_error(env, "Failed to create context wrapper");
  }

  err = js_add_type_tag(env, result, &llama_context_type_tag);
  assert(err == 0);

  return result;
}

// freeContext(ctx: Context): void
static js_value_t *
fn_free_context(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return NULL;

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_context_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid context");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return NULL;

  llama_free(ctx);

  js_value_t *null_val;
  js_get_null(env, &null_val);
  return null_val;
}

// createSampler(params?: object): Sampler
static js_value_t *
fn_create_sampler(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  struct llama_sampler *sampler = llama_sampler_chain_init(sparams);

  // Default: greedy sampling
  float temp = 0.0f;
  int32_t top_k = 40;
  float top_p = 0.95f;

  if (argc >= 1) {
    js_value_t *opts = argv[0];
    js_value_t *val;
    bool has_prop;

    err = js_has_named_property(env, opts, "temp", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "temp", &val);
      if (err == 0) {
        double d;
        js_get_value_double(env, val, &d);
        temp = (float)d;
      }
    }

    err = js_has_named_property(env, opts, "topK", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "topK", &val);
      if (err == 0) {
        js_get_value_int32(env, val, &top_k);
      }
    }

    err = js_has_named_property(env, opts, "topP", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "topP", &val);
      if (err == 0) {
        double d;
        js_get_value_double(env, val, &d);
        top_p = (float)d;
      }
    }
  }

  // Build sampler chain
  if (temp > 0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(0));
  } else {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  }

  js_value_t *result;
  err = js_create_external(env, sampler, finalize_sampler, NULL, &result);
  if (err < 0) {
    llama_sampler_free(sampler);
    return throw_error(env, "Failed to create sampler wrapper");
  }

  err = js_add_type_tag(env, result, &llama_sampler_type_tag);
  assert(err == 0);

  return result;
}

// freeSampler(sampler: Sampler): void
static js_value_t *
fn_free_sampler(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return NULL;

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_sampler_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid sampler");

  struct llama_sampler *sampler;
  err = js_get_value_external(env, argv[0], (void **)&sampler);
  if (err < 0 || !sampler) return NULL;

  llama_sampler_free(sampler);

  js_value_t *null_val;
  js_get_null(env, &null_val);
  return null_val;
}

// tokenize(model: Model, text: string, addBos: boolean): Int32Array
static js_value_t *
fn_tokenize(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 3;
  js_value_t *argv[3];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 2) return throw_error(env, "Model and text required");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_model_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid model");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  // Get text
  size_t text_len;
  err = js_get_value_string_utf8(env, argv[1], NULL, 0, &text_len);
  if (err < 0) return throw_error(env, "Invalid text");

  char *text = malloc(text_len + 1);
  if (!text) return throw_error(env, "Memory allocation failed");

  err = js_get_value_string_utf8(env, argv[1], (utf8_t *)text, text_len + 1, NULL);
  if (err < 0) {
    free(text);
    return throw_error(env, "Failed to read text");
  }

  bool add_bos = true;
  if (argc >= 3) {
    js_get_value_bool(env, argv[2], &add_bos);
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(model);

  // Estimate token count (generous)
  int32_t max_tokens = text_len + 16;
  llama_token *tokens = malloc(max_tokens * sizeof(llama_token));
  if (!tokens) {
    free(text);
    return throw_error(env, "Memory allocation failed");
  }

  int32_t n_tokens = llama_tokenize(vocab, text, text_len, tokens, max_tokens, add_bos, true);
  free(text);

  if (n_tokens < 0) {
    // Need more space
    max_tokens = -n_tokens;
    tokens = realloc(tokens, max_tokens * sizeof(llama_token));
    if (!tokens) return throw_error(env, "Memory allocation failed");
    n_tokens = llama_tokenize(vocab, text, text_len, tokens, max_tokens, add_bos, true);
  }

  if (n_tokens < 0) {
    free(tokens);
    return throw_error(env, "Tokenization failed");
  }

  // Create Int32Array
  js_value_t *array_buffer;
  void *data;
  err = js_create_arraybuffer(env, n_tokens * sizeof(int32_t), &data, &array_buffer);
  if (err < 0) {
    free(tokens);
    return throw_error(env, "Failed to create array buffer");
  }

  memcpy(data, tokens, n_tokens * sizeof(int32_t));
  free(tokens);

  js_value_t *result;
  err = js_create_typedarray(env, js_int32array, n_tokens, array_buffer, 0, &result);
  if (err < 0) return throw_error(env, "Failed to create typed array");

  return result;
}

// detokenize(model: Model, tokens: Int32Array): string
static js_value_t *
fn_detokenize(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 2) return throw_error(env, "Model and tokens required");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_model_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid model");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  // Get tokens array
  bool is_typedarray;
  err = js_is_typedarray(env, argv[1], &is_typedarray);
  if (err < 0 || !is_typedarray) return throw_error(env, "Tokens must be Int32Array");

  js_typedarray_type_t type;
  size_t length;
  void *data;
  err = js_get_typedarray_info(env, argv[1], &type, &data, &length, NULL, NULL);
  if (err < 0 || type != js_int32array) return throw_error(env, "Tokens must be Int32Array");

  llama_token *tokens = (llama_token *)data;

  const struct llama_vocab *vocab = llama_model_get_vocab(model);

  // Build output string
  size_t buf_size = length * 16;  // Estimate
  char *buf = malloc(buf_size);
  if (!buf) return throw_error(env, "Memory allocation failed");

  size_t offset = 0;
  for (size_t i = 0; i < length; i++) {
    char piece[256];
    int32_t n = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, true);
    if (n > 0) {
      if (offset + n >= buf_size) {
        buf_size *= 2;
        buf = realloc(buf, buf_size);
        if (!buf) return throw_error(env, "Memory allocation failed");
      }
      memcpy(buf + offset, piece, n);
      offset += n;
    }
  }

  js_value_t *result;
  err = js_create_string_utf8(env, (utf8_t *)buf, offset, &result);
  free(buf);

  if (err < 0) return throw_error(env, "Failed to create string");

  return result;
}

// decode(ctx: Context, tokens: Int32Array): void
static js_value_t *
fn_decode(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 2) return throw_error(env, "Context and tokens required");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_context_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid context");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

  // Get tokens
  bool is_typedarray;
  err = js_is_typedarray(env, argv[1], &is_typedarray);
  if (err < 0 || !is_typedarray) return throw_error(env, "Tokens must be Int32Array");

  js_typedarray_type_t type;
  size_t length;
  void *data;
  err = js_get_typedarray_info(env, argv[1], &type, &data, &length, NULL, NULL);
  if (err < 0 || type != js_int32array) return throw_error(env, "Tokens must be Int32Array");

  struct llama_batch batch = llama_batch_get_one((llama_token *)data, length);

  int decode_result = llama_decode(ctx, batch);
  if (decode_result != 0) {
    return throw_error(env, "Decode failed");
  }

  js_value_t *undefined;
  js_get_undefined(env, &undefined);
  return undefined;
}

// sample(ctx: Context, sampler: Sampler, idx: number): number
static js_value_t *
fn_sample(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 3;
  js_value_t *argv[3];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 3) return throw_error(env, "Context, sampler, and index required");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_context_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid context");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

  err = js_check_type_tag(env, argv[1], &llama_sampler_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid sampler");

  struct llama_sampler *sampler;
  err = js_get_value_external(env, argv[1], (void **)&sampler);
  if (err < 0 || !sampler) return throw_error(env, "Invalid sampler");

  int32_t idx;
  err = js_get_value_int32(env, argv[2], &idx);
  if (err < 0) return throw_error(env, "Invalid index");

  llama_token token = llama_sampler_sample(sampler, ctx, idx);

  js_value_t *result;
  err = js_create_int32(env, token, &result);
  if (err < 0) return throw_error(env, "Failed to create result");

  return result;
}

// acceptToken(sampler: Sampler, token: number): void
static js_value_t *
fn_accept_token(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_sampler_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid sampler");

  struct llama_sampler *sampler;
  err = js_get_value_external(env, argv[0], (void **)&sampler);
  if (err < 0 || !sampler) return throw_error(env, "Invalid sampler");

  int32_t token;
  js_get_value_int32(env, argv[1], &token);

  llama_sampler_accept(sampler, token);

  js_value_t *undefined;
  js_get_undefined(env, &undefined);
  return undefined;
}

// isEogToken(model: Model, token: number): boolean
static js_value_t *
fn_is_eog_token(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  bool valid;
  err = js_check_type_tag(env, argv[0], &llama_model_type_tag, &valid);
  if (err < 0 || !valid) return throw_error(env, "Invalid model");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  int32_t token;
  js_get_value_int32(env, argv[1], &token);

  const struct llama_vocab *vocab = llama_model_get_vocab(model);
  bool is_eog = llama_vocab_is_eog(vocab, token);

  js_value_t *result;
  js_get_boolean(env, is_eog, &result);
  return result;
}

// Finalizers
static void finalize_model(js_env_t *env, void *data, void *hint) {
  (void)env; (void)hint;
  if (data) llama_model_free((struct llama_model *)data);
}

static void finalize_context(js_env_t *env, void *data, void *hint) {
  (void)env; (void)hint;
  if (data) llama_free((struct llama_context *)data);
}

static void finalize_sampler(js_env_t *env, void *data, void *hint) {
  (void)env; (void)hint;
  if (data) llama_sampler_free((struct llama_sampler *)data);
}

// Helper macro for defining functions
#define EXPORT_FUNCTION(name, fn) do { \
  js_value_t *func; \
  err = js_create_function(env, name, -1, fn, NULL, &func); \
  assert(err == 0); \
  err = js_set_named_property(env, exports, name, func); \
  assert(err == 0); \
} while (0)

static js_value_t *
addon_exports(js_env_t *env, js_value_t *exports) {
  int err;

  // Initialize llama backend
  llama_backend_init();

  EXPORT_FUNCTION("loadModel", fn_load_model);
  EXPORT_FUNCTION("freeModel", fn_free_model);
  EXPORT_FUNCTION("createContext", fn_create_context);
  EXPORT_FUNCTION("freeContext", fn_free_context);
  EXPORT_FUNCTION("createSampler", fn_create_sampler);
  EXPORT_FUNCTION("freeSampler", fn_free_sampler);
  EXPORT_FUNCTION("tokenize", fn_tokenize);
  EXPORT_FUNCTION("detokenize", fn_detokenize);
  EXPORT_FUNCTION("decode", fn_decode);
  EXPORT_FUNCTION("sample", fn_sample);
  EXPORT_FUNCTION("acceptToken", fn_accept_token);
  EXPORT_FUNCTION("isEogToken", fn_is_eog_token);

  return exports;
}

BARE_MODULE(llama, addon_exports)
