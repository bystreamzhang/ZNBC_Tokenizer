const byId = (id) => document.getElementById(id);

const elements = {
  form: byId("experiment-form"),
  corpus: byId("corpus-input"),
  vocab: byId("vocab-input"),
  protected: byId("protected-input"),
  text: byId("text-input"),
  run: byId("run-button"),
  status: byId("request-status"),
  source: byId("source-label"),
  error: byId("error-message"),
  results: byId("results"),
  metricVocab: byId("metric-vocab"),
  metricVocabNote: byId("metric-vocab-note"),
  metricMerges: byId("metric-merges"),
  metricPieces: byId("metric-pieces"),
  metricPiecesNote: byId("metric-pieces-note"),
  metricCompression: byId("metric-compression"),
  metricCompressionNote: byId("metric-compression-note"),
  policyId: byId("policy-id"),
  protectedList: byId("protected-list"),
  protectedOmittedNote: byId("protected-omitted-note"),
  protectedRaw: byId("protected-raw"),
  protectedPattern: byId("protected-pattern"),
  categoryPattern: byId("category-pattern"),
  policyRules: byId("policy-rules"),
  trainingSamples: byId("training-samples"),
  trainingCountLabel: byId("training-count-label"),
  trainingOmittedNote: byId("training-omitted-note"),
  mergeRows: byId("merge-rows"),
  noMerges: byId("no-merges"),
  mergeCountLabel: byId("merge-count-label"),
  mergeOmittedNote: byId("merge-omitted-note"),
  encodingRatio: byId("encoding-ratio"),
  encodingPieces: byId("encoding-pieces"),
  encodingOmittedNote: byId("encoding-omitted-note"),
  tokenOutput: byId("token-output"),
  decodedOutput: byId("decoded-output"),
  invariantList: byId("invariant-list"),
};

elements.protected.value = " .,!?:;|\t\n。";
let requestSerial = 0;

const MAX_VISIBLE_PROTECTED_CHARACTERS = 20;
const MAX_VISIBLE_TRAINING_SAMPLES = 3;
const MAX_VISIBLE_PIECES_PER_SAMPLE = 16;
const MAX_VISIBLE_MERGES = 24;
const MAX_VISIBLE_ENCODING_PIECES = 64;

const invariantLabels = {
  split_rebuilds_inputs: "pieces 无损拼回原文",
  protected_pieces_unchanged: "protected pieces 从未 merge",
  pieces_flatten_to_encoding: "piece tokens 正确展平",
  encoded_bytes_match_input: "输出 bytes 等于输入",
  decode_round_trip: "decode(encode(text)) round-trip",
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

function formatRatio(value) {
  return Number(value).toFixed(2) + "×";
}

function formatPercent(value) {
  return (Number(value) * 100).toFixed(1) + "%";
}

function renderPiece(piece) {
  const card = make(
    "article",
    "piece" + (piece.merge_allowed ? "" : " piece--protected"),
  );
  const top = make("div", "piece__top");
  const text = make("span", "piece__text", piece.display || "∅");
  text.title = piece.text;
  top.append(
    text,
    make(
      "span",
      "piece__kind",
      piece.merge_allowed ? piece.kind : "protected",
    ),
  );

  const meta = make(
    "div",
    "piece__meta",
    "[" + piece.start + ", " + piece.end + ") · bytes [" +
      piece.utf8_bytes.join(", ") + "]",
  );
  const tokens = make("div", "piece__tokens");
  const initial = make(
    "span",
    "",
    "[" + piece.initial_tokens.join(", ") + "]",
  );
  const arrow = document.createTextNode(" → ");
  const final = make(
    "strong",
    "",
    "[" + piece.tokens.join(", ") + "]",
  );
  tokens.append(initial, arrow, final);

  card.append(top, meta, tokens);
  return card;
}

function renderPieceStrip(container, pieces, maxVisible) {
  clear(container);
  if (pieces.length === 0) {
    container.append(
      make("p", "empty-state", "空 string 没有 split pieces。"),
    );
    return 0;
  }
  const visiblePieces = pieces.slice(0, maxVisible);
  for (const piece of visiblePieces) container.append(renderPiece(piece));
  return pieces.length - visiblePieces.length;
}

function renderSummary(result) {
  const training = result.training;
  elements.metricVocab.textContent = String(training.actual_vocab_size);
  elements.metricVocabNote.textContent =
    "请求 " + training.requested_vocab_size + "；边界可能让训练提前结束";
  elements.metricMerges.textContent = String(training.merges_learned);
  elements.metricPieces.textContent = String(training.split_piece_count);
  elements.metricPiecesNote.textContent =
    training.protected_piece_count + " protected · " +
    training.mergeable_piece_count + " mergeable";
  elements.metricCompression.textContent =
    formatRatio(training.compression_ratio);
  elements.metricCompressionNote.textContent =
    training.original_token_count + " → " + training.final_token_count +
    " · 减少 " + formatPercent(training.reduction_ratio);
}

function renderPolicy(result) {
  const policy = result.split_policy;
  elements.policyId.textContent = policy.id;
  elements.protectedPattern.textContent =
    policy.protected_pattern ||
    "disabled（没有 configured protected character）";
  elements.categoryPattern.textContent = policy.category_pattern;
  elements.protectedRaw.textContent =
    "raw config: " +
    JSON.stringify(result.configuration.protected_characters) +
    " · normalization: " + policy.normalization;

  clear(elements.protectedList);
  if (policy.protected_characters.length === 0) {
    elements.protectedList.append(
      make(
        "span",
        "muted",
        "没有 protected code point；类别边界仍然生效。",
      ),
    );
  } else {
    const visibleCharacters = policy.protected_characters.slice(
      0,
      MAX_VISIBLE_PROTECTED_CHARACTERS,
    );
    for (const character of visibleCharacters) {
      const pill = make(
        "span",
        "character-pill",
        character.display + " · " + character.codepoint,
      );
      pill.title = character.name;
      elements.protectedList.append(pill);
    }
  }
  const omittedProtected = Math.max(
    0,
    policy.protected_characters.length - MAX_VISIBLE_PROTECTED_CHARACTERS,
  );
  elements.protectedOmittedNote.hidden = omittedProtected === 0;
  elements.protectedOmittedNote.textContent =
    omittedProtected === 0 ? "" : "另有 " + omittedProtected + " 个字符未展示。";

  clear(elements.policyRules);
  for (const rule of policy.rules) {
    const card = make(
      "article",
      "rule-card" +
        (rule.merge_allowed ? "" : " rule-card--protected"),
    );
    card.append(
      make(
        "strong",
        "",
        rule.kind + " · merge " +
          (rule.merge_allowed ? "yes" : "never"),
      ),
      make("span", "", rule.description),
    );
    elements.policyRules.append(card);
  }
}

function renderTrainingSamples(samples) {
  clear(elements.trainingSamples);
  const visibleSamples = samples.slice(0, MAX_VISIBLE_TRAINING_SAMPLES);
  elements.trainingCountLabel.textContent = samples.length + " samples";
  visibleSamples.forEach((sample) => {
    const block = make("article", "sample");
    const header = make("div", "sample__header");
    const title = make(
      "strong",
      "",
      "sample " + (sample.sample_index + 1),
    );
    title.title = sample.text;
    header.append(
      title,
      make(
        "span",
        "",
        sample.initial_token_count + " byte ids → " +
          sample.final_token_count + " tokens",
      ),
    );
    const strip = make("div", "piece-strip");
    const omittedPieces = renderPieceStrip(
      strip,
      sample.pieces,
      MAX_VISIBLE_PIECES_PER_SAMPLE,
    );
    block.append(header, strip);
    if (omittedPieces > 0) {
      block.append(
        make(
          "p",
          "omitted-note",
          "这个 sample 另有 " + omittedPieces + " 个 pieces 未展示。",
        ),
      );
    }
    elements.trainingSamples.append(block);
  });

  const omittedSamples = samples.length - visibleSamples.length;
  elements.trainingOmittedNote.hidden = omittedSamples === 0;
  elements.trainingOmittedNote.textContent =
    omittedSamples === 0
      ? ""
      : "另有 " + omittedSamples + " 个训练 samples 未展示。";
}

function renderMerges(merges) {
  clear(elements.mergeRows);
  const visibleMerges = merges.slice(0, MAX_VISIBLE_MERGES);
  elements.mergeCountLabel.textContent = merges.length + " rules";
  elements.noMerges.hidden = merges.length !== 0;

  for (const merge of visibleMerges) {
    const row = document.createElement("tr");
    const values = [
      "#" + merge.rank,
      "[" + merge.pair.join(", ") + "]",
      String(merge.token_id),
      String(merge.training_frequency),
    ];
    values.forEach((value, index) => {
      const cell = document.createElement("td");
      if (index === 1 || index === 2) {
        cell.append(make("code", "", value));
      } else {
        cell.textContent = value;
      }
      row.append(cell);
    });
    const content = document.createElement("td");
    content.append(
      make("code", "", "[" + merge.bytes.join(", ") + "]"),
      document.createTextNode(" · " + JSON.stringify(merge.text)),
    );
    row.append(content);
    elements.mergeRows.append(row);
  }

  const omittedMerges = merges.length - visibleMerges.length;
  elements.mergeOmittedNote.hidden = omittedMerges === 0;
  elements.mergeOmittedNote.textContent =
    omittedMerges === 0
      ? ""
      : "仅展示前 " + visibleMerges.length + " 条，另有 " +
        omittedMerges + " 条未展示。";
}

function renderEncoding(encoding) {
  const omittedPieces = renderPieceStrip(
    elements.encodingPieces,
    encoding.pieces,
    MAX_VISIBLE_ENCODING_PIECES,
  );
  elements.encodingOmittedNote.hidden = omittedPieces === 0;
  elements.encodingOmittedNote.textContent =
    omittedPieces === 0
      ? ""
      : "另有 " + omittedPieces + " 个 encode pieces 未展示；token ids 保持完整。";
  elements.encodingRatio.textContent =
    encoding.utf8_byte_count + " bytes → " + encoding.token_count +
    " tokens · " + formatRatio(encoding.compression_ratio);
  elements.tokenOutput.textContent =
    JSON.stringify(encoding.tokens, null, 2);
  elements.decodedOutput.textContent = encoding.decoded_text;
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
    result.source.class + " · schema v" + result.schema_version;
  renderSummary(result);
  renderPolicy(result);
  renderTrainingSamples(result.training_samples);
  renderMerges(result.merges);
  renderEncoding(result.encoding);
  renderInvariants(result.invariants);
  elements.results.hidden = false;
}

async function requestOverview(payload) {
  const response = await fetch("/api/split-bpe/overview", {
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
  const payload = {
    corpus: elements.corpus.value.split("\n"),
    vocab_size: Number(elements.vocab.value),
    protected_characters: elements.protected.value,
    text: elements.text.value,
  };

  for (const control of [
    elements.corpus,
    elements.vocab,
    elements.protected,
    elements.text,
    elements.run,
  ]) {
    control.disabled = true;
  }
  elements.error.hidden = true;
  elements.results.hidden = true;
  elements.status.textContent = "正在 split、训练并编码…";

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
      for (const control of [
        elements.corpus,
        elements.vocab,
        elements.protected,
        elements.text,
        elements.run,
      ]) {
        control.disabled = false;
      }
    }
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  void run();
});

void run();
