const byId = (id) => document.getElementById(id);

const elements = {
  form: byId("tokenizer-form"),
  text: byId("text-input"),
  run: byId("run-button"),
  status: byId("request-status"),
  error: byId("error-message"),
  results: byId("results"),
  source: byId("source-label"),
  metricModel: byId("metric-model"),
  metricEncoding: byId("metric-encoding"),
  metricVocab: byId("metric-vocab"),
  metricBytes: byId("metric-bytes"),
  metricTokens: byId("metric-tokens"),
  metricBytesPerToken: byId("metric-bytes-per-token"),
  metricTokensPerByte: byId("metric-tokens-per-byte"),
  encodingSummary: byId("encoding-summary"),
  tokenList: byId("token-list"),
  tokenOmittedNote: byId("token-omitted-note"),
  tokenIdsOutput: byId("token-ids-output"),
  decodedOutput: byId("decoded-output"),
  invariantList: byId("invariant-list"),
};

const MAX_VISIBLE_TOKENS = 300;
let requestSerial = 0;

const invariantLabels = {
  token_bytes_match_input: "token bytes 无损拼回输入 bytes",
  decode_round_trip: "decode(encode(text)) 等于原文",
};

function clear(node) {
  node.replaceChildren();
}

function make(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatDecimal(value) {
  return Number(value).toFixed(3);
}

function renderMetrics(result) {
  const tokenizer = result.tokenizer;
  const metrics = result.metrics;
  elements.metricModel.textContent = tokenizer.model;
  elements.metricEncoding.textContent = tokenizer.encoding;
  elements.metricVocab.textContent = String(tokenizer.vocab_size);
  elements.metricBytes.textContent = String(metrics.utf8_byte_count);
  elements.metricTokens.textContent = String(metrics.token_count);
  elements.metricBytesPerToken.textContent = formatDecimal(
    metrics.bytes_per_token,
  );
  elements.metricTokensPerByte.textContent =
    formatDecimal(metrics.tokens_per_byte) + " tokens / byte";
  elements.encodingSummary.textContent =
    metrics.utf8_byte_count + " bytes → " + metrics.token_count + " tokens";
}

function renderToken(token) {
  const card = make("article", "token-card");
  const heading = make("div", "token-card__heading");
  heading.append(
    make("span", "token-card__index", "#" + token.index),
    make("code", "token-card__id", "id " + token.token_id),
  );

  const display = make(
    "pre",
    "token-card__display",
    token.display === "" ? "∅" : token.display,
  );
  display.title = "Python 返回的安全 UTF-8 显示";

  const metadata = make("dl", "token-card__metadata");
  const entries = [
    ["Byte offset", "[" + token.byte_start + ", " + token.byte_end + ")"],
    ["Bytes", "[" + token.bytes.join(", ") + "]"],
    ["Hex", token.bytes_hex || "∅"],
  ];
  for (const [label, value] of entries) {
    metadata.append(
      make("dt", "", label),
      make("dd", "", value),
    );
  }

  card.append(heading, display, metadata);
  return card;
}

function renderTokens(encoding) {
  clear(elements.tokenList);
  const tokens = encoding.tokens;
  const visibleTokens = tokens.slice(0, MAX_VISIBLE_TOKENS);
  if (tokens.length === 0) {
    elements.tokenList.append(
      make("p", "empty-state", "空文本没有 token。"),
    );
  } else {
    for (const token of visibleTokens) {
      elements.tokenList.append(renderToken(token));
    }
  }

  const omittedCount = tokens.length - visibleTokens.length;
  elements.tokenOmittedNote.hidden = omittedCount === 0;
  elements.tokenOmittedNote.textContent =
    omittedCount === 0
      ? ""
      : "逐 token 详情最多展示前 " + MAX_VISIBLE_TOKENS +
        " 项，另有 " + omittedCount + " 项未展示；下方 token ids 保持完整。";

  elements.tokenIdsOutput.textContent = JSON.stringify(encoding.ids, null, 2);
}

function renderInvariants(invariants) {
  clear(elements.invariantList);
  for (const [key, passed] of Object.entries(invariants)) {
    elements.invariantList.append(
      make(
        "span",
        "check" + (passed ? "" : " check--failed"),
        (passed ? "✓ " : "× ") + (invariantLabels[key] || key),
      ),
    );
  }
}

function render(result) {
  elements.source.textContent =
    result.source.class + "." + result.source.method +
    " · schema v" + result.schema_version;
  renderMetrics(result);
  renderTokens(result.encoding);
  elements.decodedOutput.textContent = result.encoding.decoded_text;
  renderInvariants(result.invariants);
  elements.results.hidden = false;
}

async function requestOverview(payload) {
  const response = await fetch("/api/tiktoken-gpt-4/overview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "server 返回了无法解析的响应（HTTP " + response.status + "）",
    );
  }
  if (!response.ok) {
    throw new Error(
      data.error || "请求失败（HTTP " + response.status + "）",
    );
  }
  return data;
}

async function run() {
  const currentSerial = ++requestSerial;
  const payload = { text: elements.text.value };

  elements.text.disabled = true;
  elements.run.disabled = true;
  elements.error.hidden = true;
  elements.results.hidden = true;
  elements.status.textContent = "正在调用 GPT4Tokenizer.analyze…";

  try {
    const result = await requestOverview(payload);
    if (currentSerial !== requestSerial) return;
    render(result);
    elements.status.textContent = "完成";
  } catch (error) {
    if (currentSerial !== requestSerial) return;
    elements.error.textContent =
      error instanceof Error ? error.message : String(error);
    elements.error.hidden = false;
    elements.status.textContent = "运行失败";
  } finally {
    if (currentSerial === requestSerial) {
      elements.text.disabled = false;
      elements.run.disabled = false;
    }
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  void run();
});

void run();
