#!/bin/bash
JOBS=${1:-4}
RESULTS_FILE=$(mktemp)
echo "Running tests..."
run_test() {
    TEST="$1"
    RESULTS_FILE="$2"
    NAME=$(basename "$TEST")
    OUTPUT=$(./attn_seq_first "$TEST" 2>&1)
    if [ $? -eq 0 ]; then
        echo "✓ $NAME"
        echo "PASS" >> "$RESULTS_FILE"
    else
        SUMMARY=$(echo "$OUTPUT" | grep "SUMMARY:" | head -1)
        echo "✗ $NAME - $SUMMARY"
        echo "FAIL:$NAME" >> "$RESULTS_FILE"
    fi
}
export -f run_test
cat << 'TESTLIST' | xargs -P "$JOBS" -I {} bash -c 'run_test "{}" "'$RESULTS_FILE'"'
tests_seq_first/randn_s1_h1_c2_sh0.txt
tests_seq_first/randn_s1_h1_c2_sh0_t96.txt
tests_seq_first/randn_s1_h1_c2_sh1.txt
tests_seq_first/randn_s1_h1_c2_sh1_t96.txt
tests_seq_first/randn_s1_h1_c4_sh0.txt
tests_seq_first/randn_s1_h1_c4_sh0_t224.txt
tests_seq_first/randn_s1_h1_c4_sh2.txt
tests_seq_first/randn_s1_h1_c4_sh2_t224.txt
tests_seq_first/randn_s1_h4_c2_sh0.txt
tests_seq_first/randn_s1_h4_c2_sh0_t96.txt
tests_seq_first/randn_s1_h4_c2_sh1.txt
tests_seq_first/randn_s1_h4_c2_sh1_t96.txt
tests_seq_first/randn_s1_h4_c4_sh0.txt
tests_seq_first/randn_s1_h4_c4_sh0_t224.txt
tests_seq_first/randn_s1_h4_c4_sh2.txt
tests_seq_first/randn_s1_h4_c4_sh2_t224.txt
tests_seq_first/randn_s4_h1_c2_sh0.txt
tests_seq_first/randn_s4_h1_c2_sh0_t96.txt
tests_seq_first/randn_s4_h1_c2_sh1.txt
tests_seq_first/randn_s4_h1_c2_sh1_t96.txt
tests_seq_first/randn_s4_h1_c4_sh0.txt
tests_seq_first/randn_s4_h1_c4_sh0_t224.txt
tests_seq_first/randn_s4_h1_c4_sh2.txt
tests_seq_first/randn_s4_h1_c4_sh2_t224.txt
tests_seq_first/randn_s4_h4_c2_sh0.txt
tests_seq_first/randn_s4_h4_c2_sh0_t96.txt
tests_seq_first/randn_s4_h4_c2_sh1.txt
tests_seq_first/randn_s4_h4_c2_sh1_t96.txt
tests_seq_first/randn_s4_h4_c4_sh0.txt
tests_seq_first/randn_s4_h4_c4_sh0_t224.txt
tests_seq_first/randn_s4_h4_c4_sh2.txt
tests_seq_first/randn_s4_h4_c4_sh2_t224.txt
tests_seq_first/small_s1_h1_c2_sh0.txt
tests_seq_first/small_s1_h1_c2_sh0_t96.txt
tests_seq_first/small_s1_h1_c2_sh1.txt
tests_seq_first/small_s1_h1_c2_sh1_t96.txt
tests_seq_first/small_s1_h1_c4_sh0.txt
tests_seq_first/small_s1_h1_c4_sh0_t224.txt
tests_seq_first/small_s1_h1_c4_sh2.txt
tests_seq_first/small_s1_h1_c4_sh2_t224.txt
tests_seq_first/small_s1_h4_c2_sh0.txt
tests_seq_first/small_s1_h4_c2_sh0_t96.txt
tests_seq_first/small_s1_h4_c2_sh1.txt
tests_seq_first/small_s1_h4_c2_sh1_t96.txt
tests_seq_first/small_s1_h4_c4_sh0.txt
tests_seq_first/small_s1_h4_c4_sh0_t224.txt
tests_seq_first/small_s1_h4_c4_sh2.txt
tests_seq_first/small_s1_h4_c4_sh2_t224.txt
tests_seq_first/small_s4_h1_c2_sh0.txt
tests_seq_first/small_s4_h1_c2_sh0_t96.txt
tests_seq_first/small_s4_h1_c2_sh1.txt
tests_seq_first/small_s4_h1_c2_sh1_t96.txt
tests_seq_first/small_s4_h1_c4_sh0.txt
tests_seq_first/small_s4_h1_c4_sh0_t224.txt
tests_seq_first/small_s4_h1_c4_sh2.txt
tests_seq_first/small_s4_h1_c4_sh2_t224.txt
tests_seq_first/small_s4_h4_c2_sh0.txt
tests_seq_first/small_s4_h4_c2_sh0_t96.txt
tests_seq_first/small_s4_h4_c2_sh1.txt
tests_seq_first/small_s4_h4_c2_sh1_t96.txt
tests_seq_first/small_s4_h4_c4_sh0.txt
tests_seq_first/small_s4_h4_c4_sh0_t224.txt
tests_seq_first/small_s4_h4_c4_sh2.txt
tests_seq_first/small_s4_h4_c4_sh2_t224.txt
TESTLIST
# Count logic same as before
PASSED=$(grep -c "^PASS$" "$RESULTS_FILE" || true)
FAILED=0
FAILED_TESTS=""
while IFS= read -r line; do
    if [[ "$line" == FAIL:* ]]; then
        ((FAILED++))
        NAME="${line#FAIL:}"
        FAILED_TESTS="$FAILED_TESTS
  $NAME"
    fi
done < <(grep "^FAIL:" "$RESULTS_FILE" 2>/dev/null || true)
rm -f "$RESULTS_FILE"
echo ""
echo "PASSED: $PASSED / $((PASSED + FAILED))"
if [ "$FAILED" -gt 0 ]; then
    echo -e "FAILED:$FAILED_TESTS"
    exit 1
else
    echo "All tests passed!"
fi
