const byId = (id) => document.getElementById(id);

const elements = {
  form: byId("experiment-form"),
  mode: byId("mode-input"),
  specialPolicy: byId("special-policy-input"),
  trainingConfig: byId("training-config"),
  trainingText: byId("training-text-input"),
  vocab: byId("vocab-input"),
  text: byId("text-input"),
  run: byId("run-button"),
  status: byId("request-status"),
  error: byId("error-message"),
  results: byId("results"),
  source: byId("source-label"),
  summaryDescription: byId("summary-description"),
  metricMode: byId("metric-mode"),
  metricVocab: byId("metric-vocab"),
  metricMerges: byId("metric-merges"),
  metricBytes: byId("metric-bytes"),
  metricTokens: byId("metric-tokens"),
  metricRatio: byId("metric-ratio"),
  specialTokenList: byId("special-token-list"),
  pattern: byId("pattern-output"),
  pieceList: byId("piece-list"),
  piecesSummary: byId("pieces-summary"),
  pieceOmitted: byId("piece-omitted-note"),
  mergeBody: byId("merge-table-body"),
  mergesSummary: byId("merges-summary"),
  mergeOmitted: byId("merge-omitted-note"),
  tokenList: byId("token-list"),
  encodingSummary: byId("encoding-summary"),
  tokenOmitted: byId("token-omitted-note"),
  tokenIds: byId("token-ids-output"),
  referenceStage: byId("reference-stage"),
  referenceIds: byId("reference-ids-output"),
  referenceChecks: byId("reference-checks"),
  decoded: byId("decoded-output"),
  invariants: byId("invariant-list"),
};

const MAX_VISIBLE_PIECES = 120;
const MAX_VISIBLE_MERGES = 160;
const MAX_VISIBLE_TOKENS = 300;
let requestSerial = 0;

const invariantLabels = {
  pieces_rebuild_input: "regex pieces 连续拼回普通输入",
  token_bytes_match_input: "token bytes 无损拼回输入 bytes",
  decode_round_trip: "decode(encode(text)) 等于原文",
  own_ids_match_tiktoken: "自研 token ids 与 tiktoken 一致",
  own_decode_matches_tiktoken: "自研 decode 与 tiktoken 一致",
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

function updateMode() {
  const training = elements.mode.value === "train";
  elements.trainingConfig.hidden = !training;
  elements.run.textContent = training ? "训练并运行 tokenizer" : "运行固定 GPT-4 tokenizer";
}

function renderChecks(node, checks) {
  clear(node);
  for (const [key, passed] of Object.entries(checks)) {
    node.append(
      make(
        "span",
        "check" + (passed ? "" : " check--failed"),
        (passed ? "✓ " : "× ") + (invariantLabels[key] || key),
      ),
    );
  }
}

function renderSummary(result) {
  const config = result.configuration;
  const metrics = result.metrics;
  elements.source.textContent =
    result.source.class + " · schema v" + result.schema_version;
  elements.summaryDescription.textContent =
    config.mode === "gpt4"
      ? "固定加载 cl100k_base 的 merge ranks 与 byte permutation。"
      : "使用同一个 GPT-4 regex pattern 完成训练和编码。";
  elements.metricMode.textContent = config.mode === "gpt4" ? "GPT-4" : "Train";
  elements.metricVocab.textContent = String(config.vocab_size);
  elements.metricMerges.textContent = String(config.merge_count);
  elements.metricBytes.textContent = String(metrics.utf8_byte_count);
  elements.metricTokens.textContent = String(metrics.token_count);
  elements.metricRatio.textContent = formatDecimal(metrics.bytes_per_token);

  clear(elements.specialTokenList);
  for (const [literal, tokenId] of Object.entries(config.special_tokens)) {
    elements.specialTokenList.append(
      make("code", "special-pill", literal + " → " + tokenId),
    );
  }
}

function renderPieces(result) {
  const pieces = result.pieces;
  const visible = pieces.slice(0, MAX_VISIBLE_PIECES);
  clear(elements.pieceList);
  elements.pattern.textContent = result.configuration.pattern;
  elements.piecesSummary.textContent = pieces.length + " pieces";
  if (pieces.length === 0) {
    elements.pieceList.append(make("p", "empty-state", "空文本没有 regex piece。"));
  } else {
    for (const piece of visible) {
      const card = make(
        "article",
        "piece-card" + (piece.kind === "special" ? " piece-card--special" : ""),
      );
      card.append(
        make("span", "piece-index", "#" + piece.index + " · " + piece.kind),
        make("pre", "piece-text", piece.text === "" ? "∅" : piece.display),
        make("code", "piece-meta", "chars [" + piece.char_start + ", " + piece.char_end + ") · " + piece.utf8_byte_count + " bytes"),
      );
      elements.pieceList.append(card);
    }
  }
  const omitted = pieces.length - visible.length;
  elements.pieceOmitted.hidden = omitted === 0;
  elements.pieceOmitted.textContent = omitted
    ? "最多展示前 " + MAX_VISIBLE_PIECES + " 个 pieces，另有 " + omitted + " 个未展示。"
    : "";
}

function renderMerges(result) {
  const merges = result.merges;
  const visible = merges.slice(0, MAX_VISIBLE_MERGES);
  clear(elements.mergeBody);
  for (const row of visible) {
    const tr = document.createElement("tr");
    tr.append(
      make("td", "mono-cell", String(row.rank)),
      make("td", "mono-cell", "(" + row.pair.join(", ") + ")"),
      make("td", "mono-cell", String(row.token_id)),
      make("td", "merge-value", row.bytes_hex + (row.display ? " · " + row.display : "")),
    );
    elements.mergeBody.append(tr);
  }
  elements.mergesSummary.textContent = result.configuration.merge_count + " total";
  const omitted = result.configuration.merge_count - visible.length;
  elements.mergeOmitted.hidden = omitted <= 0;
  elements.mergeOmitted.textContent = omitted > 0
    ? "后端只返回/前端只展示前 " + visible.length + " 条，另有 " + omitted + " 条未展示。"
    : "";
}

function renderToken(token) {
  const card = make(
    "article",
    "token-card" + (token.kind === "special" ? " token-card--special" : ""),
  );
  const heading = make("div", "token-card__heading");
  heading.append(
    make("span", "token-card__index", "#" + token.index),
    make("span", "token-kind", token.kind),
    make("code", "token-card__id", "id " + token.token_id),
  );
  const metadata = make("dl", "token-card__metadata");
  const entries = [
    ["Display", token.display === "" ? "∅" : token.display],
    ["Byte offset", "[" + token.byte_start + ", " + token.byte_end + ")"],
    ["Bytes", "[" + token.bytes.join(", ") + "]"],
    ["Hex", token.bytes_hex || "∅"],
  ];
  for (const [label, value] of entries) {
    metadata.append(make("dt", "", label), make("dd", "", value));
  }
  card.append(heading, metadata);
  return card;
}

function renderEncoding(result) {
  const encoding = result.encoding;
  const visible = encoding.tokens.slice(0, MAX_VISIBLE_TOKENS);
  clear(elements.tokenList);
  if (encoding.tokens.length === 0) {
    elements.tokenList.append(make("p", "empty-state", "空文本没有 token。"));
  } else {
    for (const token of visible) elements.tokenList.append(renderToken(token));
  }
  const omitted = encoding.tokens.length - visible.length;
  elements.tokenOmitted.hidden = omitted === 0;
  elements.tokenOmitted.textContent = omitted
    ? "逐 token 明细最多展示前 " + MAX_VISIBLE_TOKENS + " 项，另有 " + omitted + " 项未展示；完整 ids 保留在下方。"
    : "";
  elements.encodingSummary.textContent =
    result.metrics.utf8_byte_count + " bytes → " + result.metrics.token_count + " tokens";
  elements.tokenIds.textContent = JSON.stringify(encoding.ids, null, 2);
  elements.decoded.textContent = encoding.decoded_text;
}

function renderReference(result) {
  const reference = result.reference;
  elements.referenceStage.hidden = reference === null;
  if (reference === null) return;
  elements.referenceIds.textContent = JSON.stringify(reference.ids, null, 2);
  renderChecks(elements.referenceChecks, {
    own_ids_match_tiktoken: reference.ids_match,
    own_decode_matches_tiktoken: reference.decode_match,
  });
}

function render(result) {
  renderSummary(result);
  renderPieces(result);
  renderMerges(result);
  renderEncoding(result);
  renderReference(result);
  renderChecks(elements.invariants, result.invariants);
  elements.results.hidden = false;
}

async function requestOverview(payload) {
  const response = await fetch("/api/own-gpt4/overview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("server 返回了无法解析的响应（HTTP " + response.status + "）");
  }
  if (!response.ok) {
    throw new Error(data.error || "请求失败（HTTP " + response.status + "）");
  }
  return data;
}

async function run() {
  const currentSerial = ++requestSerial;
  const mode = elements.mode.value;
  const payload = {
    mode,
    special_policy: elements.specialPolicy.value,
    text: elements.text.value,
  };
  if (mode === "train") {
    payload.training_text = elements.trainingText.value;
    payload.vocab_size = Number(elements.vocab.value);
  }

  for (const control of [elements.mode, elements.specialPolicy, elements.trainingText, elements.vocab, elements.text, elements.run]) {
    control.disabled = true;
  }
  elements.error.hidden = true;
  elements.results.hidden = true;
  elements.status.textContent = mode === "gpt4" ? "正在恢复并运行 cl100k_base…" : "正在训练 RegexTokenizer…";

  try {
    const result = await requestOverview(payload);
    if (currentSerial !== requestSerial) return;
    render(result);
    elements.status.textContent = "完成";
  } catch (error) {
    if (currentSerial !== requestSerial) return;
    elements.error.textContent = error instanceof Error ? error.message : String(error);
    elements.error.hidden = false;
    elements.status.textContent = "运行失败";
  } finally {
    if (currentSerial === requestSerial) {
      for (const control of [elements.mode, elements.specialPolicy, elements.trainingText, elements.vocab, elements.text, elements.run]) {
        control.disabled = false;
      }
    }
  }
}

elements.mode.addEventListener("change", updateMode);
elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  void run();
});

updateMode();
void run();
