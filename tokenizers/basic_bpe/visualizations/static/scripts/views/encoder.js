import { requestBpeEncode } from "../api.js";
import {
  formatPercentage,
  formatRatio,
  makeInvariant,
  renderSequences,
} from "../ui.js";

function requiredElement(selector) {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`页面缺少必要元素：${selector}`);
  return element;
}

function formatTokenList(tokens) {
  return JSON.stringify(tokens);
}

export function initializeEncoderView({ onResult = null } = {}) {
  const elements = {
    form: requiredElement("#encoder-form"),
    text: requiredElement("#encoder-text-input"),
    runButton: requiredElement("#encode-button"),
    status: requiredElement("#encoder-request-status"),
    error: requiredElement("#encoder-error-message"),
    context: requiredElement("#encoder-context"),
    source: requiredElement("#encoder-source-label"),
    results: requiredElement("#encoder-results"),
    metricBytes: requiredElement("#encoder-metric-bytes"),
    metricTokens: requiredElement("#encoder-metric-tokens"),
    metricReduction: requiredElement("#encoder-metric-reduction"),
    metricRules: requiredElement("#encoder-metric-rules"),
    metricApplied: requiredElement("#encoder-metric-applied"),
    metricOperations: requiredElement("#encoder-metric-operations"),
    previous: requiredElement("#encoder-previous-step"),
    next: requiredElement("#encoder-next-step"),
    range: requiredElement("#encoder-step-range"),
    stepLabel: requiredElement("#encoder-step-label"),
    stepCount: requiredElement("#encoder-step-count"),
    mergeRule: requiredElement("#encoder-merge-rule"),
    ruleNote: requiredElement("#encoder-rule-note"),
    before: requiredElement("#encoder-before-sequence"),
    after: requiredElement("#encoder-after-sequence"),
    outputRatio: requiredElement("#encoder-output-ratio"),
    outputList: requiredElement("#encoder-output-list"),
    finalTokens: requiredElement("#encoder-final-tokens"),
    invariants: requiredElement("#encoder-invariant-checks"),
  };

  let tokenizerContext = null;
  let result = null;
  let stepIndex = 0;
  let requestSerial = 0;

  function showError(message) {
    elements.error.textContent = message;
    elements.error.hidden = false;
  }

  function clearError() {
    elements.error.textContent = "";
    elements.error.hidden = true;
  }

  function payloadFromContext() {
    return {
      corpus: [...tokenizerContext.corpus],
      vocab_size: tokenizerContext.vocabSize,
      text: elements.text.value,
    };
  }

  function renderSummary() {
    const encoding = result.encoding;
    const trace = result.trace;
    elements.source.textContent = `${result.source.module} · ${result.source.class}.${result.source.method}()`;
    elements.metricBytes.textContent = String(encoding.utf8_byte_count);
    elements.metricTokens.textContent = String(encoding.token_count);
    elements.metricReduction.textContent = `减少 ${formatPercentage(encoding.reduction_ratio)}`;
    elements.metricRules.textContent = String(trace.rules_checked);
    elements.metricApplied.textContent = String(trace.rules_applied);
    elements.metricOperations.textContent = `共合并 ${trace.merge_operations} 对 token`;
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
      elements.stepLabel.textContent = "没有 learned merge rule";
      elements.stepCount.textContent = `${result.encoding.utf8_byte_count} → ${result.encoding.token_count} tokens`;
      elements.mergeRule.textContent = "当前 Tokenizer 只包含 256 个基础 byte token。";
      elements.ruleNote.textContent = "Encoder 直接返回输入 string 的 UTF-8 byte ids。";
      renderSequences(
        elements.before,
        [result.encoding.initial_tokens],
        result.vocabulary,
        { labels: ["UTF-8 初始序列"] },
      );
      renderSequences(
        elements.after,
        [result.encoding.tokens],
        result.vocabulary,
        { labels: ["最终序列"] },
      );
      return;
    }

    const step = steps[stepIndex];
    elements.range.disabled = false;
    elements.range.min = "1";
    elements.range.max = String(steps.length);
    elements.range.value = String(stepIndex + 1);
    elements.previous.disabled = stepIndex === 0;
    elements.next.disabled = stepIndex === steps.length - 1;
    elements.stepLabel.textContent = `规则 ${step.rule_number} / ${steps.length}`;
    elements.stepCount.textContent = `${step.before.length} → ${step.after.length} tokens`;
    elements.mergeRule.textContent = `(${step.pair[0]}, ${step.pair[1]}) → ${step.token_id} · 本次命中 ${step.applied_merge_count} 次`;

    if (!step.applied) {
      elements.ruleNote.textContent = "本条固定规则在当前序列中没有命中；Encoder 不改变顺序，继续检查下一条规则。";
    } else if (step.depends_on_previous_token) {
      elements.ruleNote.textContent = "这条规则使用了前序规则生成的 token，展示了合并结果继续触发后续 merge 的级联。";
    } else {
      elements.ruleNote.textContent = "本条规则按训练时确定的 rank 执行；训练频率不会参与本次编码决策。";
    }

    renderSequences(
      elements.before,
      [step.before],
      result.vocabulary,
      { mergedPairStarts: [step.merged_pair_starts], labels: ["当前序列"] },
    );
    renderSequences(
      elements.after,
      [step.after],
      result.vocabulary,
      { newTokenId: step.token_id, labels: ["规则执行后"] },
    );
  }

  function renderOutput() {
    const encoding = result.encoding;
    elements.outputRatio.textContent = `${encoding.utf8_byte_count} bytes → ${encoding.token_count} tokens · ${formatRatio(encoding.compression_ratio)}`;
    elements.outputList.textContent = formatTokenList(encoding.tokens);
    renderSequences(
      elements.finalTokens,
      [encoding.tokens],
      result.vocabulary,
      { labels: [encoding.text ? `encode(${JSON.stringify(encoding.text)})` : "encode(空 string)"] },
    );
  }

  function renderInvariants() {
    elements.invariants.replaceChildren(
      makeInvariant(
        "merge token id 与训练顺序一致",
        result.invariants.rules_follow_training_order,
      ),
      makeInvariant(
        "每条规则只依赖更早的 token id",
        result.invariants.parents_precede_children,
      ),
      makeInvariant(
        "逐规则 trace 与核心 encode() 一致",
        result.invariants.trace_matches_encoder,
      ),
      makeInvariant(
        "输出 token 对应的 bytes 可还原输入",
        result.invariants.encoded_bytes_match_input,
      ),
    );
  }

  function renderAll() {
    renderSummary();
    renderStep();
    renderOutput();
    renderInvariants();
    elements.results.hidden = false;
  }

  async function run() {
    clearError();
    if (!tokenizerContext) {
      showError("请先在 BPE 构建页训练一个 Tokenizer。");
      return;
    }

    const payload = payloadFromContext();
    const currentSerial = ++requestSerial;
    elements.runButton.disabled = true;
    elements.status.textContent = "正在按固定 merge 顺序编码…";
    try {
      const nextResult = await requestBpeEncode(payload);
      // 如果较新的请求已经发出，就丢弃旧请求晚到的响应。
      if (currentSerial !== requestSerial) return;
      result = nextResult;
      stepIndex = 0;
      renderAll();
      if (onResult) onResult(result, payload);
      elements.status.textContent = "完成";
    } catch (error) {
      if (currentSerial !== requestSerial) return;
      showError(error instanceof Error ? error.message : String(error));
      elements.status.textContent = "编码失败";
    } finally {
      if (currentSerial === requestSerial) elements.runButton.disabled = false;
    }
  }

  function useTokenizerResult(trainingResult, payload) {
    tokenizerContext = {
      corpus: [...payload.corpus],
      vocabSize: payload.vocab_size,
    };
    elements.context.textContent = `已接收 BPE 构建结果：实际词表 ${trainingResult.training.actual_vocab_size}，按顺序保存 ${trainingResult.training.merges_learned} 条 merge rule。待编码 string 不会修改它。`;
    void run();
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

  return { useTokenizerResult };
}
