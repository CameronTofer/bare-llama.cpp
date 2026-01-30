#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <bare.h>
#include <js.h>
#include <utf.h>
#include <llama.h>
#include "sampling.h"
#include "log.h"

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

  char *path = (char *)malloc(path_len + 1);
  if (!path) return throw_error(env, "Memory allocation failed");

  err = js_get_value_string_utf8(env, argv[0], (utf8_t *)path, path_len + 1, NULL);
  if (err < 0) {
    free(path);
    return throw_error(env, "Failed to read model path");
  }

  // Set up default params
  struct llama_model_params params = llama_model_default_params();
  params.progress_callback = NULL;  // Disable progress callback
  // use_mmap defaults to true - keep it for better memory usage

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

  // Note: js_add_type_tag crashes in Bare runtime, so we skip type tagging

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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

    // embeddings
    err = js_has_named_property(env, opts, "embeddings", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "embeddings", &val);
      if (err == 0) {
        bool embd;
        js_get_value_bool(env, val, &embd);
        params.embeddings = embd;
      }
    }

    // poolingType (0=unspecified, 1=none, 2=mean, 3=cls, 4=last, 5=rank)
    err = js_has_named_property(env, opts, "poolingType", &has_prop);
    if (err == 0 && has_prop) {
      err = js_get_named_property(env, opts, "poolingType", &val);
      if (err == 0) {
        int32_t n;
        js_get_value_int32(env, val, &n);
        params.pooling_type = (enum llama_pooling_type)n;
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

  // Skip type tagging - js_add_type_tag crashes in Bare runtime
  // err = js_add_type_tag(env, result, &llama_context_type_tag);

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

  // Skip type check - js_add_type_tag crashes in Bare runtime
  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return NULL;

  llama_free(ctx);

  js_value_t *null_val;
  js_get_null(env, &null_val);
  return null_val;
}

// clearMemory(ctx: Context): void - Clear context memory for reuse
static js_value_t *
fn_clear_memory(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

  llama_memory_t mem = llama_get_memory(ctx);
  if (mem) {
    llama_memory_clear(mem, true);
  }

  js_value_t *undefined;
  js_get_undefined(env, &undefined);
  return undefined;
}

// Helper to get string property
static char *get_string_property(js_env_t *env, js_value_t *opts, const char *name) {
  int err;
  bool has_prop;
  js_value_t *val;

  err = js_has_named_property(env, opts, name, &has_prop);
  if (err != 0 || !has_prop) return NULL;

  err = js_get_named_property(env, opts, name, &val);
  if (err != 0) return NULL;

  size_t len;
  err = js_get_value_string_utf8(env, val, NULL, 0, &len);
  if (err != 0) return NULL;

  char *str = (char *)malloc(len + 1);
  if (!str) return NULL;

  err = js_get_value_string_utf8(env, val, (utf8_t *)str, len + 1, NULL);
  if (err != 0) {
    free(str);
    return NULL;
  }

  return str;
}

// createSampler(model: Model, params?: object): Sampler
static js_value_t *
fn_create_sampler(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 1) return throw_error(env, "Model required");

  // Get model for vocab access (needed for grammar)
  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  const struct llama_vocab *vocab = llama_model_get_vocab(model);

  struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
  struct llama_sampler *sampler = llama_sampler_chain_init(sparams);

  // Default: greedy sampling
  float temp = 0.0f;
  int32_t top_k = 40;
  float top_p = 0.95f;

  // Grammar options (llguidance)
  char *json_grammar = NULL;
  char *lark_grammar = NULL;

  if (argc >= 2) {
    js_value_t *opts = argv[1];
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

    // Grammar options (llguidance): json or lark
    json_grammar = get_string_property(env, opts, "json");
    lark_grammar = get_string_property(env, opts, "lark");
  }

  // Add grammar sampler first if specified (must filter logits before sampling)
  if (json_grammar) {
    struct llama_sampler *grammar = llama_sampler_init_llg(vocab, "json", json_grammar);
    if (grammar) {
      llama_sampler_chain_add(sampler, grammar);
    }
    free(json_grammar);
  } else if (lark_grammar) {
    struct llama_sampler *grammar = llama_sampler_init_llg(vocab, "lark", lark_grammar);
    if (grammar) {
      llama_sampler_chain_add(sampler, grammar);
    }
    free(lark_grammar);
  }

  // Build sampler chain (after grammar filtering)
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

  // Skip type tagging - js_add_type_tag crashes in Bare runtime
  // err = js_add_type_tag(env, result, &llama_sampler_type_tag);

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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

  // Skip type check - js_add_type_tag crashes in Bare runtime
  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  // Get text
  size_t text_len;
  err = js_get_value_string_utf8(env, argv[1], NULL, 0, &text_len);
  if (err < 0) return throw_error(env, "Invalid text");

  char *text = (char *)malloc(text_len + 1);
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
  llama_token *tokens = (llama_token *)malloc(max_tokens * sizeof(llama_token));
  if (!tokens) {
    free(text);
    return throw_error(env, "Memory allocation failed");
  }

  int32_t n_tokens = llama_tokenize(vocab, text, text_len, tokens, max_tokens, add_bos, true);
  free(text);

  if (n_tokens < 0) {
    // Need more space
    max_tokens = -n_tokens;
    tokens = (llama_token *)realloc(tokens, max_tokens * sizeof(llama_token));
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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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
  char *buf = (char *)malloc(buf_size);
  if (!buf) return throw_error(env, "Memory allocation failed");

  size_t offset = 0;
  for (size_t i = 0; i < length; i++) {
    char piece[256];
    int32_t n = llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece), 0, true);
    if (n > 0) {
      if (offset + n >= buf_size) {
        buf_size *= 2;
        buf = (char *)realloc(buf, buf_size);
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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

  // Skip type checks - js_add_type_tag crashes in Bare runtime
  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

  // Skip type check - js_add_type_tag crashes in Bare runtime
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

// getEmbeddingDimension(model: Model): number
static js_value_t *
fn_get_embedding_dimension(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  int32_t n_embd = llama_model_n_embd(model);

  js_value_t *result;
  err = js_create_int32(env, n_embd, &result);
  if (err < 0) return throw_error(env, "Failed to create result");

  return result;
}

// getTrainingContextSize(model: Model): number
static js_value_t *
fn_get_training_context_size(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  struct llama_model *model;
  err = js_get_value_external(env, argv[0], (void **)&model);
  if (err < 0 || !model) return throw_error(env, "Invalid model");

  int32_t n_ctx_train = llama_model_n_ctx_train(model);

  js_value_t *result;
  err = js_create_int32(env, n_ctx_train, &result);
  if (err < 0) return throw_error(env, "Failed to create result");

  return result;
}

// getContextSize(ctx: Context): number
static js_value_t *
fn_get_context_size(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

  uint32_t n_ctx = llama_n_ctx(ctx);

  js_value_t *result;
  err = js_create_uint32(env, n_ctx, &result);
  if (err < 0) return throw_error(env, "Failed to create result");

  return result;
}

// getEmbeddings(ctx: Context, idx: number): Float32Array
// idx: sequence ID for pooled embeddings, or token index for non-pooled
static js_value_t *
fn_get_embeddings(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 2;
  js_value_t *argv[2];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return throw_error(env, "Failed to get callback info");

  if (argc < 2) return throw_error(env, "Context and index required");

  struct llama_context *ctx;
  err = js_get_value_external(env, argv[0], (void **)&ctx);
  if (err < 0 || !ctx) return throw_error(env, "Invalid context");

  int32_t idx;
  err = js_get_value_int32(env, argv[1], &idx);
  if (err < 0) return throw_error(env, "Invalid index");

  // Try sequence embeddings first (for pooled embeddings)
  // then fall back to token embeddings
  float *embeddings = llama_get_embeddings_seq(ctx, idx >= 0 ? idx : 0);
  if (!embeddings) {
    embeddings = llama_get_embeddings_ith(ctx, idx);
  }
  if (!embeddings) return throw_error(env, "Failed to get embeddings (context may not have embeddings enabled)");

  const struct llama_model *model = llama_get_model(ctx);
  int32_t n_embd = llama_model_n_embd(model);

  // Create Float32Array
  js_value_t *array_buffer;
  void *data;
  err = js_create_arraybuffer(env, n_embd * sizeof(float), &data, &array_buffer);
  if (err < 0) return throw_error(env, "Failed to create array buffer");

  memcpy(data, embeddings, n_embd * sizeof(float));

  js_value_t *result;
  err = js_create_typedarray(env, js_float32array, n_embd, array_buffer, 0, &result);
  if (err < 0) return throw_error(env, "Failed to create typed array");

  return result;
}

// Log level control
static int g_log_level = 2;  // 0=off, 1=errors only, 2=all (default)

static void quiet_log_callback(enum ggml_log_level level, const char *text, void *user_data) {
  (void)user_data;
  (void)level;
  // Debug: uncomment to see what's being logged
  // fprintf(stderr, "[LOG %d/%d] %s", g_log_level, level, text);
  if (g_log_level == 0) return;
  if (g_log_level == 1) {
    // In errors-only mode, suppress llguidance completion messages
    if (strncmp(text, "llg error:", 10) == 0) return;
    if (level > GGML_LOG_LEVEL_ERROR) return;
  }
  fprintf(stderr, "%s", text);
}

// setLogLevel(level: number): void  - 0=off, 1=errors, 2=all
static js_value_t *
fn_set_log_level(js_env_t *env, js_callback_info_t *info) {
  int err;
  size_t argc = 1;
  js_value_t *argv[1];

  err = js_get_callback_info(env, info, &argc, argv, NULL, NULL);
  if (err < 0) return NULL;

  int32_t level = 2;
  if (argc >= 1) {
    js_get_value_int32(env, argv[0], &level);
  }

  g_log_level = level;
  llama_log_set(quiet_log_callback, NULL);

  // Also control common library logging
  if (level == 0) {
    common_log_pause(common_log_main());
  } else {
    common_log_resume(common_log_main());
  }

  js_value_t *undefined;
  js_get_undefined(env, &undefined);
  return undefined;
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

  // Initialize llama backend and set default log callback
  llama_backend_init();
  llama_log_set(quiet_log_callback, NULL);

  EXPORT_FUNCTION("loadModel", fn_load_model);
  EXPORT_FUNCTION("freeModel", fn_free_model);
  EXPORT_FUNCTION("createContext", fn_create_context);
  EXPORT_FUNCTION("freeContext", fn_free_context);
  EXPORT_FUNCTION("clearMemory", fn_clear_memory);
  EXPORT_FUNCTION("createSampler", fn_create_sampler);
  EXPORT_FUNCTION("freeSampler", fn_free_sampler);
  EXPORT_FUNCTION("tokenize", fn_tokenize);
  EXPORT_FUNCTION("detokenize", fn_detokenize);
  EXPORT_FUNCTION("decode", fn_decode);
  EXPORT_FUNCTION("sample", fn_sample);
  EXPORT_FUNCTION("acceptToken", fn_accept_token);
  EXPORT_FUNCTION("isEogToken", fn_is_eog_token);
  EXPORT_FUNCTION("getEmbeddingDimension", fn_get_embedding_dimension);
  EXPORT_FUNCTION("getTrainingContextSize", fn_get_training_context_size);
  EXPORT_FUNCTION("getContextSize", fn_get_context_size);
  EXPORT_FUNCTION("getEmbeddings", fn_get_embeddings);
  EXPORT_FUNCTION("setLogLevel", fn_set_log_level);

  return exports;
}

BARE_MODULE(llama, addon_exports)
