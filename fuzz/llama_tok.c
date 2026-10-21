// llama.cpp's tokenizer for `python -m tokcpu gguf`, as one long-lived process: the GGUF counterpart of
// compat.mjs. One request per stdin line, one answer per stdout line; text travels as hex, so any byte can:
//   load PATH        -> ok N_VOCAB | error MESSAGE      a GGUF's vocabulary only, no weights
//   encode HEX       -> ids 1 2 3                      special tokens added and parsed, as transformers does
//   decode 1 2 3     -> text HEX                       special tokens removed, as skip_special_tokens does
// Built by tokcpu/gguf.py with: cc llama_tok.c -I<llama.cpp>/include -L<llama.cpp>/lib -lllama
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

static char last_error[512];   // llama.cpp logs why a load failed; keep the last such line to answer with

static void quiet(enum ggml_log_level level, const char * text, void * data) {
    (void) data;
    if (level == GGML_LOG_LEVEL_ERROR && strstr(text, "error loading model")) {
        snprintf(last_error, sizeof last_error, "%s", text);
        last_error[strcspn(last_error, "\n")] = 0;
    }
}

static struct llama_model * model;
static const struct llama_vocab * vocab;

static int unhex(const char * s, char * out) {
    int n = 0;
    for (; s[0] && s[1] && s[0] != '\n'; s += 2) {
        unsigned v;
        sscanf(s, "%2x", &v);
        out[n++] = (char) v;
    }
    return n;
}

int main(void) {
    llama_log_set(quiet, NULL);
    llama_backend_init();
    char * line = NULL;
    size_t cap = 0;
    ssize_t len;
    while ((len = getline(&line, &cap, stdin)) > 0) {
        if (strncmp(line, "load ", 5) == 0) {
            line[strcspn(line, "\n")] = 0;
            if (model) llama_model_free(model);
            struct llama_model_params p = llama_model_default_params();
            p.vocab_only = true;
            last_error[0] = 0;
            model = llama_model_load_from_file(line + 5, p);
            vocab = model ? llama_model_get_vocab(model) : NULL;
            if (model) printf("ok %d\n", llama_vocab_n_tokens(vocab));
            else printf("error %s\n", last_error[0] ? last_error : "llama.cpp cannot load it");
        } else if (!model) {
            printf("error nothing loaded\n");
        } else if (strncmp(line, "encode ", 7) == 0) {
            char * text = malloc(len);
            int n = unhex(line + 7, text);
            int max = n + 16;
            llama_token * ids = malloc(sizeof(llama_token) * max);
            int got = llama_tokenize(vocab, text, n, ids, max, true, true);
            if (got < 0) {
                max = -got;
                ids = realloc(ids, sizeof(llama_token) * max);
                got = llama_tokenize(vocab, text, n, ids, max, true, true);
            }
            if (got < 0) printf("error tokenize failed\n");
            else {
                printf("ids");
                for (int i = 0; i < got; i++) printf(" %d", ids[i]);
                printf("\n");
            }
            free(ids);
            free(text);
        } else if (strncmp(line, "decode", 6) == 0) {
            int n = 0, max = 16;
            llama_token * ids = malloc(sizeof(llama_token) * max);
            for (char * s = strtok(line + 6, " \n"); s; s = strtok(NULL, " \n")) {
                if (n == max) ids = realloc(ids, sizeof(llama_token) * (max *= 2));
                ids[n++] = atoi(s);
            }
            int size = 64 + 16 * n;
            char * text = malloc(size);
            int got = llama_detokenize(vocab, ids, n, text, size, true, false);
            if (got < 0) {
                size = -got;
                text = realloc(text, size);
                got = llama_detokenize(vocab, ids, n, text, size, true, false);
            }
            if (got < 0) printf("error detokenize failed\n");
            else {
                printf("text ");
                for (int i = 0; i < got; i++) printf("%02x", (unsigned char) text[i]);
                printf("\n");
            }
            free(ids);
            free(text);
        } else {
            printf("error unknown request\n");
        }
        fflush(stdout);
    }
    if (model) llama_model_free(model);
    llama_backend_free();
    return 0;
}
