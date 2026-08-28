import { requestBpeTrace } from "../api.js";
import {
  formatPercentage,
  formatRatio,
  makeInvariant,
  renderPairCounts,
  renderSequences,
} from "../ui.js";

function requiredElement(selector) {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`页面缺少必要元素：${selector}`);
  return element;
}

export function initializeBpeView({ onResult = null } = {}) {
  const elements = {
    form: requiredElement("#bpe-form"),
    corpus: requiredElement("#corpus-input"),
    vocab: requiredElement("#vocab-input"),
    runButton: requiredElement("#run-button"),
    status: requiredElement("#request-status"),
    error: requiredElement("#error-message"),
    results: requiredElement("#results"),
    source: requiredElement("#source-label"),
    metricVocab: requiredElement("#metric-vocab"),
    metricVocabContext: requiredElement("#metric-vocab-context"),
    metricMerges: requiredElement("#metric-merges"),
    metricTokens: requiredElement("#metric-tokens"),
    metricTokenContext: requiredElement("#metric-token-context"),
    metricRatio: requiredElement("#metric-ratio"),
    metricReduction: requiredElement("#metric-reduction"),
    previous: requiredElement("#previous-step"),
    next: requiredElement("#next-step"),
    range: requiredElement("#step-range"),
    stepLabel: requiredElement("#step-label"),
    stepTokenCount: requiredElement("#step-token-count"),
    mergeRule: requiredElement("#merge-rule"),
    before: requiredElement("#before-sequences"),
    after: requiredElement("#after-sequences"),
    pairList: requiredElement("#pair-list"),
    pairCountNote: requiredElement("#pair-count-note"),
    invariants: requiredElement("#invariant-checks"),
  };

  let result = null;
  let stepIndex = 0;

  function payloadFromInputs() {
    return {
      corpus: elements.corpus.value.split("\n"),
      vocab_size: Number(elements.vocab.value),
      // BPE 页只负责构建；空 string 仅满足 trace API 的兼容字段。
      text: "",
    };
  }

  function showError(message) {
    elements.error.textContent = message;
    elements.error.hidden = false;
  }

  function clearError() {
    elements.error.textContent = "";
    elements.error.hidden = true;
  }

  function renderSummary() {
    const training = result.training;
    elements.source.textContent = `${result.source.module} · ${result.source.class}`;
    elements.metricVocab.textContent = String(training.actual_vocab_size);
    elements.metricVocabContext.textContent = `目标 ${training.requested_vocab_size}`;
    elements.metricMerges.textContent = String(training.merges_learned);
    elements.metricTokens.textContent = `${training.original_token_count} → ${training.final_token_count}`;
    elements.metricTokenContext.textContent = `减少 ${training.original_token_count - training.final_token_count} 个`;
    elements.metricRatio.textContent = formatRatio(training.compression_ratio);
    elements.metricReduction.textContent = `token 数减少 ${formatPercentage(training.reduction_ratio)}`;
  }

  function renderStep() {
    const steps = result.trace.steps;
    if (steps.length === 0) {
      elements.range.min = "0";
      elements.range.max = "0";
      elements.range.value = "0";
      elements.range.disabled = true;
      elements.previous.disabled = true;
      elements.next.disabled = true;
      elements.stepLabel.textContent = "没有执行 merge";
      const finalCount = result.trace.final_sequences.reduce(
        (total, tokens) => total + tokens.length,
        0,
      );
      elements.stepTokenCount.textContent = `${finalCount} → ${finalCount} tokens`;
      elements.mergeRule.textContent = "目标词表未要求 merge，或样本中没有相邻 pair。";
      renderSequences(
        elements.before,
        result.trace.initial_sequences,
        result.vocabulary,
      );
      renderSequences(
        elements.after,
        result.trace.final_sequences,
        result.vocabulary,
      );
      renderPairCounts(elements.pairList, null, result.vocabulary);
      elements.pairCountNote.textContent = "0 种 pair";
      return;
    }

    const step = steps[stepIndex];
    const beforeCount = step.before.reduce((total, tokens) => total + tokens.length, 0);
    const afterCount = step.after.reduce((total, tokens) => total + tokens.length, 0);
    elements.range.disabled = false;
    elements.range.min = "1";
    elements.range.max = String(steps.length);
    elements.range.value = String(stepIndex + 1);
    elements.previous.disabled = stepIndex === 0;
    elements.next.disabled = stepIndex === steps.length - 1;
    elements.stepLabel.textContent = `第 ${step.round} / ${steps.length} 轮`;
    elements.stepTokenCount.textContent = `${beforeCount} → ${afterCount} tokens`;
    elements.mergeRule.textContent = `(${step.pair[0]}, ${step.pair[1]}) · 频率 ${step.training_frequency} · 实际合并 ${step.applied_merge_count} 次 → token ${step.token_id}`;

    renderSequences(elements.before, step.before, result.vocabulary, {
      mergedPairStarts: step.merged_pair_starts,
    });
    renderSequences(elements.after, step.after, result.vocabulary, {
      newTokenId: step.token_id,
    });
    renderPairCounts(elements.pairList, step, result.vocabulary);
    elements.pairCountNote.textContent = step.pair_counts_truncated
      ? `共 ${step.pair_type_count} 种，显示前 ${step.pair_counts.length} 种`
      : `共 ${step.pair_type_count} 种`;
  }

  function renderInvariants() {
    elements.invariants.replaceChildren(
      makeInvariant(
        "训练后的语料与按 learned rules 重放的结果一致",
        result.invariants.trace_matches_encoder,
      ),
    );
  }

  function renderAll() {
    renderSummary();
    renderStep();
    renderInvariants();
    elements.results.hidden = false;
  }

  async function run() {
    clearError();
    elements.runButton.disabled = true;
    elements.status.textContent = "正在用当前 bpe.py 构建…";
    try {
      const payload = payloadFromInputs();
      result = await requestBpeTrace(payload);
      stepIndex = 0;
      renderAll();
      if (onResult) onResult(result, payload);
      elements.status.textContent = "完成";
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
      elements.status.textContent = "运行失败";
    } finally {
      elements.runButton.disabled = false;
    }
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    void run();
  });
  elements.previous.addEventListener("click", () => {
    if (stepIndex > 0) {
      stepIndex -= 1;
      renderStep();
    }
  });
  elements.next.addEventListener("click", () => {
    if (result && stepIndex + 1 < result.trace.steps.length) {
      stepIndex += 1;
      renderStep();
    }
  });
  elements.range.addEventListener("input", () => {
    stepIndex = Math.max(0, Number(elements.range.value) - 1);
    renderStep();
  });

  void run();
}
