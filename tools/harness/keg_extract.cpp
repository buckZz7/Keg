// keg_extract.cpp — single-pass top-k next-token log-prob extraction for the
// Keg gate, using the model's own llama.cpp build (muse-glimmer supported).
//
// Reads the tokenized eval stream (tokens.txt, one int per line) and the
// sampled positions (positions.txt, one int per line). Processes the stream in
// n_ctx windows (independent contexts), and at each sampled position extracts
// the top-k log-probs (as the model predicts that token, logits at pos-1).
// Writes JSON: {"positions": {str(pos): {token: logprob}}}.
//
// usage:
//   keg_extract --model <gguf> --tokens tokens.txt --positions positions.txt \
//       --out topk.json [--top-k N] [--n-ctx N] [--min-local-ctx N] [--n-gpu N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cmath>
#include <chrono>
#include "llama.h"

static std::vector<int> read_ints(const char * path) {
    std::vector<int> out;
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    int v;
    while (fscanf(f, "%d", &v) == 1) out.push_back(v);
    fclose(f);
    return out;
}

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o;
}

static std::string utf8_clean(const std::string & in) {
    std::string out; out.reserve(in.size());
    size_t i = 0, n = in.size();
    const std::string REPL = "\xEF\xBF\xBD"; // U+FFFD
    while (i < n) {
        unsigned char c = in[i];
        int len = 0; uint32_t cp = 0;
        if (c < 0x80) { len = 1; cp = c; }
        else if ((c & 0xE0) == 0xC0) { len = 2; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { len = 3; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { len = 4; cp = c & 0x07; }
        else { out += REPL; i++; continue; }
        if (i + len > n) { out += REPL; i++; continue; }
        bool ok = true;
        for (int k = 1; k < len; k++) {
            if ((in[i+k] & 0xC0) != 0x80) { ok = false; break; }
            cp = (cp << 6) | (in[i+k] & 0x3F);
        }
        if (!ok) { out += REPL; i++; continue; }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000)
            || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) { out += REPL; i++; continue; }
        out.append(in, i, len);
        i += len;
    }
    return out;
}

static std::string piece_of(const llama_vocab * vocab, llama_token tok) {
    char buf[64];
    int n = llama_token_to_piece(vocab, tok, buf, (int)sizeof buf, 0, false);
    if (n < 0) {
        std::vector<char> big(-n + 1);
        n = llama_token_to_piece(vocab, tok, big.data(), (int)big.size(), 0, false);
        return utf8_clean(std::string(big.data(), n));
    }
    return utf8_clean(std::string(buf, n));
}

int main(int argc, char ** argv) {
    std::string model, tokens_path, positions_path, out_path;
    int top_k = 1024, n_ctx = 8192, min_local_ctx = 2048, n_gpu = 99;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens_path = next();
        else if (a == "--positions") positions_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "--top-k") top_k = atoi(next().c_str());
        else if (a == "--n-ctx") n_ctx = atoi(next().c_str());
        else if (a == "--min-local-ctx") min_local_ctx = atoi(next().c_str());
        else if (a == "--n-gpu") n_gpu = atoi(next().c_str());
    }
    if (model.empty() || tokens_path.empty() || positions_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: keg_extract --model GGUF --tokens tokens.txt --positions positions.txt --out topk.json [opts]\n");
        return 1;
    }

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu;
    llama_model * model_p = llama_model_load_from_file(model.c_str(), mparams);
    if (!model_p) { fprintf(stderr, "failed to load model\n"); return 1; }
    auto cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;   // accept a full window in one llama_decode
    cparams.n_ubatch = 512;    // internal micro-batch keeps compute buffer small
    llama_context * ctx = llama_init_from_model(model_p, cparams);
    if (!ctx) { fprintf(stderr, "failed to init context\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model_p);
    int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<int> tokens = read_ints(tokens_path.c_str());
    std::vector<int> positions = read_ints(positions_path.c_str());
    std::map<int, int> want;
    for (int p : positions) want[p] = 1;
    int n_total = (int)tokens.size();

    FILE * out = fopen(out_path.c_str(), "w");
    if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 2; }
    fprintf(out, "{\n  \"positions\": {\n");

    llama_batch batch = llama_batch_init(n_ctx, 0, 1);
    bool first = true;
    int written = 0;
    std::vector<float> row(n_vocab);
    std::vector<int> top_idx(top_k);

    auto t0 = std::chrono::steady_clock::now();
    long long total_tokens = 0;
    for (int win_start = 0; win_start < n_total; win_start += n_ctx) {
        int n = std::min(n_ctx, n_total - win_start);
        total_tokens += n;
        llama_memory_clear(mem, true); // independent window: fresh context, pos 0..n-1
        batch.n_tokens = n;
        for (int i = 0; i < n; i++) {
            batch.token[i]   = tokens[win_start + i];
            batch.pos[i]     = i;
            batch.n_seq_id[i]= 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]  = 1;
        }
        int rc = llama_decode(ctx, batch);
        if (rc != 0) fprintf(stderr, "warning: decode rc=%d at win %d\n", rc, win_start);
        float * logits = llama_get_logits(ctx);
        for (int j = 0; j < n; j++) {
            int P = win_start + j;
            if (!want.count(P)) continue;
            if (j < min_local_ctx) continue;
            const float * r = logits + (j - 1) * n_vocab; // logits at token j-1 predict P
            for (int t = 0; t < n_vocab; t++) row[t] = r[t];
            int kk = std::min(top_k, n_vocab);
            std::vector<int> idx(n_vocab);
            for (int t = 0; t < n_vocab; t++) idx[t] = t;
            std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(),
                [&](int a, int b) { return row[a] > row[b]; });
            // log-sum-exp over full row
            float mx = *std::max_element(row.begin(), row.end());
            double lse = 0.0;
            for (int t = 0; t < n_vocab; t++) lse += std::exp((double)row[t] - mx);
            lse = mx + std::log(lse);
            if (!first) fprintf(out, ",\n");
            first = false;
            fprintf(out, "    \"%d\": {", P);
            for (int k = 0; k < kk; k++) {
                int t = idx[k];
                std::string pc = piece_of(vocab, t);
                if (k) fprintf(out, ",");
                fprintf(out, "\"%s\":%.6f", json_escape(pc).c_str(), (double)(row[t] - lse));
            }
            fprintf(out, "}");
            written++;
        }
        if (n < n_ctx) break;
    }
    fprintf(out, "\n  },\n  \"n\": %d\n}\n", written);
    fclose(out);

    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "decode tps: %.1f (%lld tokens / %.1fs)\n",
            total_tokens / (secs > 0 ? secs : 1.0), total_tokens, secs);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model_p);
    fprintf(stderr, "wrote %s: %d positions\n", out_path.c_str(), written);
    return 0;
}
