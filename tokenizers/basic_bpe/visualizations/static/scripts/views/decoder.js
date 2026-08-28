import { requestBpeDecode } from "../api.js";
import { makeInvariant, renderSequences } from "../ui.js";

const MAX_VISIBLE_BYTES = 384;

function requiredElement(selector) {
  const element = document.querySelector(selector);
  if (!element) throw new Error(`页面缺少必要元素：${selector}`);
  return element;
}

function parseTokenIds(rawValue) {
  let tokens;
  try {
    tokens = JSON.parse(rawValue);
  } catch (_error) {
    throw new Error("token 输入必须是有效列表，例如 [257, 32, 228]。");
  }
  if (!Array.isArray(tokens)) {
    throw new Error("token 输入必须是一个 int 列表。");
  }
  return tokens;
}

function formatByteStream(bytes) {
  const toHex = (value) => value.toString(16).padStart(2, "0").toUpperCase();
  if (bytes.length <= MAX_VISIBLE_BYTES) {
    return bytes.map(toHex).join(" ") || "∅";
  }

  // 很长的 byte stream 只折叠展示，不改变 server 返回的完整解码结果。
  const sideLength = MAX_VISIBLE_BYTES / 2;
  const head = bytes.slice(0, sideLength).map(toHex).join(" ");
  const tail = bytes.slice(-sideLength).map(toHex).join(" ");
  return `${head} … 中间 ${bytes.length - MAX_VISIBLE_BYTES} bytes 已折叠 … ${tail}`;
}

function formatTokenList(tokens) {
  const serialized = JSON.stringify(tokens);
  if (serialized.length <= 240) return serialized;
  return `${serialized.slice(0, 116)} … ${serialized.slice(-116)}`;
}

export function initializeDecoderView() {
  const elements = {
    form: requiredElement("#decoder-form"),
    tokenInput: requiredElement("#decoder-token-input"),
    runButton: requiredElement("#decode-button"),
    status: requiredElement("#decoder-request-status"),
    error: requiredElement("#decoder-error-message"),
    context: requiredElement("#decoder-context"),
    source: requiredElement("#decoder-source-label"),
    results: requiredElement("#decoder-results"),
    metricTokens: requiredElement("#decoder-metric-tokens"),
    metricBytes: requiredElement("#decoder-metric-bytes"),
    metricLength: requiredElement("#decoder-metric-length"),
    metricVocab: requiredElement("#decoder-metric-vocab"),
    tokenSequence: requiredElement("#decoder-token-sequence"),
    byteStream: requiredElement("#decoder-byte-stream"),
    output: requiredElement("#decoder-output"),
    replacement: requiredElement("#decoder-replacement-note"),
    canonical: requiredElement("#decoder-canonical-note"),
    invariants: requiredElement("#decoder-invariant-checks"),
  };

  let encoderContext = null;
  let requestSerial = 0;

  function showError(message) {
    elements.error.textContent = message;
    elements.error.hidden = false;
  }

  function clearError() {
    elements.error.textContent = "";
    elements.error.hidden = true;
  }

  function render(result) {
    const decoding = result.decoding;
    elements.source.textContent = `${result.source.module} · ${result.source.class}.${result.source.method}()`;
    elements.metricTokens.textContent = String(decoding.token_count);
    elements.metricBytes.textContent = String(decoding.byte_count);
    elements.metricLength.textContent = String(decoding.python_length);
    elements.metricVocab.textContent = String(result.training.actual_vocab_size);

    renderSequences(
      elements.tokenSequence,
      [decoding.tokens],
      result.vocabulary,
      { labels: ["decoder 收到的 int 列表"] },
    );
    elements.byteStream.textContent = formatByteStream(decoding.decoded_bytes);

    // JSON 字符串表示能让换行、制表符和空 string 的边界保持可见。
    elements.output.textContent = JSON.stringify(decoding.text);
    elements.replacement.classList.toggle(
      "is-used",
      decoding.used_replacement,
    );
    elements.replacement.textContent = decoding.used_replacement
      ? "检测到非法或不完整的 UTF-8；errors=\"replace\" 已插入 U+FFFD（�），decoder 继续返回 string。"
      : "本次 byte stream 是有效 UTF-8，没有插入替代字符。";
    elements.canonical.classList.toggle(
      "is-noncanonical",
      !decoding.is_canonical_encoding,
    );
    if (decoding.is_canonical_encoding) {
      elements.canonical.textContent = "这是当前 encoder 会生成的规范 token 列表；decode 后再次 encode，id 列表保持不变。";
    } else if (decoding.used_replacement) {
      elements.canonical.textContent = `替代字符使本次转换有损；再次 encode 会得到 ${formatTokenList(decoding.reencoded_tokens)}。`;
    } else {
      elements.canonical.textContent = `可以无损解码，但不是当前 encoder 的规范 token 列表；再次 encode 会得到 ${formatTokenList(decoding.reencoded_tokens)}。`;
    }
    elements.invariants.replaceChildren(
      makeInvariant(
        "核心 decode 结果符合 UTF-8 errors=replace 策略",
        result.invariants.decode_matches_replace_policy,
      ),
    );
    elements.results.hidden = false;
  }

  async function run() {
    clearError();
    if (!encoderContext) {
      showError("请先在 Encoder 页运行一次编码。");
      return;
    }

    let tokens;
    try {
      tokens = parseTokenIds(elements.tokenInput.value);
    } catch (error) {
      showError(error instanceof Error ? error.message : String(error));
      return;
    }

    const currentSerial = ++requestSerial;
    elements.runButton.disabled = true;
    elements.status.textContent = "正在调用 Python decoder…";
    try {
      const result = await requestBpeDecode({
        corpus: encoderContext.corpus,
        vocab_size: encoderContext.vocabSize,
        tokens,
      });
      // 如果较新的请求已经发出，就丢弃旧请求晚到的响应。
      if (currentSerial !== requestSerial) return;
      render(result);
      elements.status.textContent = "完成";
    } catch (error) {
      if (currentSerial !== requestSerial) return;
      showError(error instanceof Error ? error.message : String(error));
      elements.status.textContent = "解码失败";
    } finally {
      if (currentSerial === requestSerial) elements.runButton.disabled = false;
    }
  }

  function useEncoderResult(result, payload) {
    encoderContext = {
      corpus: [...payload.corpus],
      vocabSize: payload.vocab_size,
    };
    elements.tokenInput.value = JSON.stringify(result.encoding.tokens);
    elements.context.textContent = `已接收 Encoder 最近一次运行：实际词表 ${result.training.actual_vocab_size}，${result.training.merges_learned} 条 merge；待解码 id 来自 “${result.encoding.text || "空 string"}”。`;
    void run();
  }

  elements.form.addEventListener("submit", (event) => {
    event.preventDefault();
    void run();
  });

  return { useEncoderResult };
}
