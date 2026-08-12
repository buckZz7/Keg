// keg_sample.cpp — tokenize the eval stream and produce stratified long-context
// token positions for the Keg single-pass gate, using the model's own llama.cpp
// build (muse-glimmer supported). Loads vocab-only (fast; no weights).
//
// Emits:
//   tokens.txt      — stream token ids, one per line
//   positions.txt   — sampled token indices, one per line
//   components.json — {str(position): "prose"|"code"|"multilingual"|"technical"}
//
// Positions are token indices with >= min_local_ctx tokens of in-window context
// (valid region of each n_ctx window), sampled per component with a floor so
// every component is represented (anti-overfit).
//
// usage:
//   keg_sample --model GGUF --corpus production.txt --components production.components.txt \
//       --outdir DIR [--n N] [--n-ctx N] [--min-local-ctx N] [--min-per-component N]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <random>
#include <cmath>
#include <sys/stat.h>
#include "llama.h"

static std::string piece_of(const llama_vocab * vocab, llama_token tok) {
    char buf[64];
    int n = llama_token_to_piece(vocab, tok, buf, (int)sizeof buf, 0, false);
    if (n < 0) {
        std::vector<char> big(-n + 1);
        n = llama_token_to_piece(vocab, tok, big.data(), (int)big.size(), 0, false);
        return std::string(big.data(), n);
    }
    return std::string(buf, n);
}

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> out;
    FILE * f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    char * line = nullptr; size_t cap = 0; ssize_t n;
    while ((n = getline(&line, &cap, f)) != -1) {
        if (n > 0 && line[n-1] == '\n') n--;
        std::string s(line, n);
        if (!s.empty() && s.back() == '\r') s.pop_back();
        if (!s.empty()) out.push_back(s);
    }
    free(line); fclose(f);
    return out;
}

int main(int argc, char ** argv) {
    std::string model, corpus_path, comp_path, outdir;
    int n = 5000, n_ctx = 8192, min_local_ctx = 2048, min_per_comp = 200;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--model") model = next();
        else if (a == "--corpus") corpus_path = next();
        else if (a == "--components") comp_path = next();
        else if (a == "--outdir") outdir = next();
        else if (a == "--n") n = atoi(next().c_str());
        else if (a == "--n-ctx") n_ctx = atoi(next().c_str());
        else if (a == "--min-local-ctx") min_local_ctx = atoi(next().c_str());
        else if (a == "--min-per-component") min_per_comp = atoi(next().c_str());
    }
    if (model.empty() || corpus_path.empty() || comp_path.empty() || outdir.empty()) {
        fprintf(stderr, "usage: keg_sample --model GGUF --corpus F --components F --outdir D [opts]\n");
        return 1;
    }

    std::vector<std::string> docs = read_lines(corpus_path.c_str());
    std::vector<std::string> comps = read_lines(comp_path.c_str());
    if (comps.size() != docs.size()) {
        fprintf(stderr, "%zu comps vs %zu docs\n", comps.size(), docs.size());
        return 1;
    }
    std::string stream;
    for (size_t i = 0; i < docs.size(); i++) {
        if (i) stream += "\n\n";
        stream += docs[i];
    }

    // doc char ranges
    std::vector<std::pair<long,long>> ranges;
    long start = 0;
    for (const auto & d : docs) {
        ranges.push_back({start, start + (long)d.size()});
        start += (long)d.size() + 2;
    }
    auto doc_of = [&](long char_) {
        int lo = 0, hi = (int)ranges.size();
        while (lo < hi) { int mid = (lo + hi) / 2; if (char_ < ranges[mid].first) hi = mid; else lo = mid + 1; }
        return std::max(0, lo - 1);
    };

    llama_backend_init();
    auto mparams = llama_model_default_params();
    mparams.vocab_only = true; // only need the tokenizer
    llama_model * model_p = llama_model_load_from_file(model.c_str(), mparams);
    if (!model_p) { fprintf(stderr, "failed to load model\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model_p);

    // tokenize stream (add_special=true -> BOS, matching generation)
    std::vector<llama_token> tokens;
    {
        int need = llama_tokenize(vocab, stream.c_str(), (int)stream.size(), nullptr, 0, true, false);
        int cap = std::abs(need);
        tokens.resize(cap);
        int got = llama_tokenize(vocab, stream.c_str(), (int)stream.size(), tokens.data(), cap, true, false);
        tokens.resize(got > 0 ? got : cap);
    }
    fprintf(stderr, "stream: %zu docs, %ld chars, %zu tokens\n", docs.size(), (long)stream.size(), tokens.size());

    // token -> approx char offset via piece accumulation (skip leading BOS token)
    std::vector<long> token_start(tokens.size());
    long char_ = 0;
    long stream_len = (long)stream.size();
    for (size_t t = 0; t < tokens.size(); t++) {
        token_start[t] = std::min(char_, stream_len);
        if (t > 0) char_ += (long)piece_of(vocab, tokens[t]).size();
    }
    fprintf(stderr, "char accumulation: final char_=%ld stream_len=%ld\n", char_, stream_len);
    auto token_comp = [&](int t) -> std::string {
        int d = doc_of(token_start[t]);
        d = std::max(0, std::min(d, (int)comps.size() - 1));
        return comps[d];
    };

    // candidate positions (valid window region) by component
    std::map<std::string, std::vector<int>> cand;
    for (int P = 0; P < (int)tokens.size(); P++) {
        int local = P % n_ctx;
        if (local < min_local_ctx) continue;
        cand[token_comp(P)].push_back(P);
    }
    fprintf(stderr, "stage: candidates built\n");
    long total_cand = 0;
    for (auto & kv : cand) total_cand += (long)kv.second.size();
    fprintf(stderr, "stage: total_cand=%ld\n", total_cand);

    std::mt19937 rng(0x5eed);
    std::vector<int> selected;
    std::map<std::string,int> comp_counts;
    for (auto & kv : cand) {
        const std::string & c = kv.first;
        std::vector<int> bucket = kv.second;
        int k = std::max(min_per_comp, (int)std::lround((double)n * bucket.size() / total_cand));
        std::shuffle(bucket.begin(), bucket.end(), rng);
        k = std::min(k, (int)bucket.size());
        for (int i = 0; i < k; i++) { selected.push_back(bucket[i]); comp_counts[c]++; }
    }
    // cap to n
    if ((int)selected.size() > n) {
        int surplus = (int)selected.size() - n;
        std::vector<int> drop;
        for (auto & kv : cand) {
            int over = comp_counts[kv.first] - min_per_comp;
            if (over > 0) {
                std::vector<int> keep;
                for (int P : selected) if (token_comp(P) == kv.first) keep.push_back(P);
                int cnt = 0;
                for (int P : keep) { if (cnt < over) drop.push_back(P); cnt++; }
            }
        }
        std::shuffle(drop.begin(), drop.end(), rng);
        if (surplus < (int)drop.size()) drop.resize(surplus);
        std::map<int,int> drop_set;
        for (int p : drop) drop_set[p] = 1;
        std::vector<int> kept;
        for (int p : selected) if (!drop_set.count(p)) kept.push_back(p);
        selected = kept;
    }
    std::sort(selected.begin(), selected.end());

    std::string toks_path = outdir + "/tokens.txt";
    std::string pos_path  = outdir + "/positions.txt";
    std::string comp_path_out = outdir + "/components.json";
    mkdir(outdir.c_str(), 0755);  // create outdir if missing
    { FILE * f = fopen(toks_path.c_str(), "w"); for (auto t : tokens) fprintf(f, "%d\n", t); fclose(f); }
    { FILE * f = fopen(pos_path.c_str(), "w"); for (int p : selected) fprintf(f, "%d\n", p); fclose(f); }
    { FILE * f = fopen(comp_path_out.c_str(), "w");
      fprintf(f, "{\n");
      for (size_t i = 0; i < selected.size(); i++) {
          fprintf(f, "%s\"%d\":\"%s\"", i ? ",\n" : "", selected[i], token_comp(selected[i]).c_str());
      }
      fprintf(f, "\n}\n"); fclose(f); }

    fprintf(stderr, "selected %zu positions; components: ", selected.size());
    for (auto & kv : comp_counts) fprintf(stderr, "%s=%d ", kv.first.c_str(), kv.second);
    fprintf(stderr, "\nwrote %s %s %s\n", toks_path.c_str(), pos_path.c_str(), comp_path_out.c_str());
    llama_model_free(model_p);
    return 0;
}
