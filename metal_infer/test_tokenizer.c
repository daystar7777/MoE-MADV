#define TOKENIZER_IMPL
#include "tokenizer.h"
#include <stdio.h>
#include <string.h>

typedef struct {
    const char *text;
    uint32_t expected_ids[64];
    int expected_count;
} test_case;

int main(void) {
    bpe_tokenizer tok;
    if (bpe_load(&tok, "tokenizer.bin") != 0) {
        fprintf(stderr, "Failed to load tokenizer.bin\n");
        return 1;
    }

    test_case tests[] = {
        {"Hello, world!", {9419, 11, 1814, 0}, 4},
        {"Hello world", {9419, 1814}, 2},
        {"The quick brown fox jumps over the lazy dog.",
         {760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13}, 10},
        {"  multiple   spaces", {220, 5081, 256, 12258}, 4},
        {"123 + 456 = 579", {16, 17, 18, 478, 220, 19, 20, 21, 283, 220, 20, 22, 24}, 13},
        {"don't won't I'm you're", {14572, 914, 2677, 914, 353, 2688, 488, 2224}, 8},
        {"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>",
         {248045, 8678, 198, 2523, 513, 264, 10631, 17313, 13, 248046, 198, 248045, 846, 198, 9419, 0, 248046}, 17},
        // Edge cases
        {"", {0}, 0},
        {"a", {64}, 1},
        {" ", {220}, 1},
        {"\n", {198}, 1},
        {"Hello\nWorld", {9419, 198, 9833}, 3},
        {"<think>Let me think about this.</think>",
         {248068, 9764, 728, 1683, 883, 411, 13, 248069}, 8},
        {"\x63\x61\x66\xc3\xa9 r\xc3\xa9sum\xc3\xa9 na\xc3\xafve",  // café résumé naïve
         {895, 56868, 238976, 91603, 571}, 5},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;

    for (int t = 0; t < num_tests; t++) {
        uint32_t ids[4096];
        int n = bpe_encode(&tok, tests[t].text, ids, 4096);

        bool match = (n == tests[t].expected_count);
        if (match) {
            for (int i = 0; i < n; i++) {
                if (ids[i] != tests[t].expected_ids[i]) {
                    match = false;
                    break;
                }
            }
        }

        if (match) {
            printf("PASS: \"%s\" -> %d tokens\n", tests[t].text, n);
            passed++;
        } else {
            printf("FAIL: \"%s\"\n", tests[t].text);
            printf("  expected (%d): [", tests[t].expected_count);
            for (int i = 0; i < tests[t].expected_count; i++)
                printf("%s%u", i ? ", " : "", tests[t].expected_ids[i]);
            printf("]\n  got      (%d): [", n);
            for (int i = 0; i < n; i++)
                printf("%s%u", i ? ", " : "", ids[i]);
            printf("]\n");
        }
    }

    printf("\n%d/%d tests passed\n", passed, num_tests);
    bpe_free(&tok);
    return passed == num_tests ? 0 : 1;
}
