import type { TokenClassificationPipeline } from "@huggingface/transformers";

const MODEL_ID = "onnx-community/bert-small-pii-detection-ONNX";

const LABEL_PLACEHOLDER_MAP: Record<string, string> = {
  AGE: "[AGE]",
  COORDINATE: "[COORDINATE]",
  CREDIT_CARD: "[CREDIT_CARD]",
  DATE_TIME: "[DATE_TIME]",
  EMAIL_ADDRESS: "[EMAIL_ADDRESS]",
  FINANCIAL: "[FINANCIAL]",
  IBAN_CODE: "[IBAN_CODE]",
  IMEI: "[IMEI]",
  IP_ADDRESS: "[IP_ADDRESS]",
  LOCATION: "[LOCATION]",
  MAC_ADDRESS: "[MAC_ADDRESS]",
  NRP: "[NRP]",
  ORGANIZATION: "[ORGANIZATION]",
  PASSWORD: "[PASSWORD]",
  PERSON: "[PERSON]",
  PHONE_NUMBER: "[PHONE_NUMBER]",
  TITLE: "[TITLE]",
  URL: "[URL]",
  US_BANK_NUMBER: "[US_BANK_NUMBER]",
  US_DRIVER_LICENSE: "[US_DRIVER_LICENSE]",
  US_ITIN: "[US_ITIN]",
  US_LICENSE_PLATE: "[US_LICENSE_PLATE]",
  US_PASSPORT: "[US_PASSPORT]",
  US_SSN: "[US_SSN]",
};

export const SUPPORTED_LABELS = Object.keys(LABEL_PLACEHOLDER_MAP);

export type DetectionEntity = {
  label: string;
  text: string;
  start: number;
  end: number;
  score: number;
};

export type DetectionResult = {
  entities: DetectionEntity[];
  redactedText: string;
};

type RawEntity = {
  entity?: string;
  entity_group?: string;
  word?: string;
  score?: number;
  start?: number;
  end?: number;
};

let pipelinePromise: Promise<TokenClassificationPipeline> | null = null;
let pipelineInstance: TokenClassificationPipeline | null = null;

const normalizeLabel = (label: string) => label.replace(/^B-|^I-/, "");

const getPlaceholder = (label: string) =>
  LABEL_PLACEHOLDER_MAP[label] ?? `[${label}]`;

const normalizeWord = (word: string) =>
  word
    .replace(/\s*([@./:_-])\s*/g, "$1")
    .replace(/\s*([()])/g, "$1")
    .replace(/\s+([.,!?;:/)\]])/g, "$1")
    .replace(/([(])/g, "$1")
    .trim();

const resolveSpanFromWord = (text: string, word: string, fromIndex: number) => {
  const normalized = normalizeWord(word);
  if (!normalized) {
    return null;
  }

  const indexFromCursor = text.indexOf(normalized, fromIndex);
  if (indexFromCursor !== -1) {
    return {
      start: indexFromCursor,
      end: indexFromCursor + normalized.length,
    };
  }

  const indexFromStart = text.indexOf(normalized);
  if (indexFromStart !== -1) {
    return {
      start: indexFromStart,
      end: indexFromStart + normalized.length,
    };
  }

  const escaped = word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const softPattern = escaped
    .replace(/\\\s+/g, "\\s*")
    .replace(/\s+/g, "\\s*")
    .replace(/\\\-/g, "\\s*-\\s*")
    .replace(/\\\./g, "\\s*\\.\\s*")
    .replace(/\\@/g, "\\s*@\\s*");
  const regex = new RegExp(softPattern, "i");
  const sliced = text.slice(fromIndex);
  const matchFromCursor = regex.exec(sliced);
  if (matchFromCursor && typeof matchFromCursor.index === "number") {
    const start = fromIndex + matchFromCursor.index;
    const value = matchFromCursor[0];
    return { start, end: start + value.length };
  }

  const matchFromStart = regex.exec(text);
  if (matchFromStart && typeof matchFromStart.index === "number") {
    const start = matchFromStart.index;
    const value = matchFromStart[0];
    return { start, end: start + value.length };
  }

  return null;
};

const toDetectionEntity = (
  text: string,
  entity: RawEntity,
  cursor: number
): { entity: DetectionEntity | null; cursor: number } => {
  if (typeof entity.start !== "number" || typeof entity.end !== "number") {
    if (typeof entity.word !== "string") {
      return { entity: null, cursor };
    }
    const span = resolveSpanFromWord(text, entity.word, cursor);
    if (!span) {
      return { entity: null, cursor };
    }

    const label = normalizeLabel(entity.entity_group ?? entity.entity ?? "UNKNOWN");
    return {
      entity: {
        label,
        text: text.slice(span.start, span.end),
        start: span.start,
        end: span.end,
        score: typeof entity.score === "number" ? entity.score : 0,
      },
      cursor: span.end,
    };
  }
  if (entity.end <= entity.start || entity.start < 0 || entity.end > text.length) {
    return { entity: null, cursor };
  }

  const label = normalizeLabel(entity.entity_group ?? entity.entity ?? "UNKNOWN");
  return {
    entity: {
      label,
      text: text.slice(entity.start, entity.end) || entity.word || "",
      start: entity.start,
      end: entity.end,
      score: typeof entity.score === "number" ? entity.score : 0,
    },
    cursor: Math.max(cursor, entity.end),
  };
};

export const isPipelineReady = () => pipelineInstance !== null;

export const ensurePIIPipeline = async () => {
  if (pipelineInstance) {
    return pipelineInstance;
  }

  if (!pipelinePromise) {
    pipelinePromise = (async () => {
      if (typeof window === "undefined") {
        throw new Error("PII detection only supports browser runtime.");
      }

      const { env, pipeline } = await import("@huggingface/transformers");
      env.allowLocalModels = false;
      env.useBrowserCache = true;

      const detector = (await pipeline("token-classification", MODEL_ID)) as TokenClassificationPipeline;
      pipelineInstance = detector;
      return detector;
    })();
  }

  try {
    return await pipelinePromise;
  } catch (error) {
    pipelinePromise = null;
    throw error;
  }
};

export const redactText = (input: string, entities: DetectionEntity[]) => {
  if (!entities.length) {
    return input;
  }

  const sorted = [...entities].sort((a, b) => {
    if (a.start !== b.start) {
      return a.start - b.start;
    }
    return b.end - a.end;
  });

  let cursor = 0;
  let output = "";

  for (const entity of sorted) {
    if (entity.start < cursor) {
      continue;
    }
    output += input.slice(cursor, entity.start);
    output += getPlaceholder(entity.label);
    cursor = entity.end;
  }

  output += input.slice(cursor);
  return output;
};

export const detectPII = async (text: string): Promise<DetectionResult> => {
  const trimmed = text.trim();
  if (!trimmed) {
    return {
      entities: [],
      redactedText: text,
    };
  }

  const classifier = await ensurePIIPipeline();
  const response = await classifier(text, {
    aggregation_strategy: "simple",
  });

  const rawEntities = Array.isArray(response) ? (response as RawEntity[]) : [response as RawEntity];
  let cursor = 0;
  const entities: DetectionEntity[] = [];
  for (const raw of rawEntities) {
    const mapped = toDetectionEntity(text, raw, cursor);
    cursor = mapped.cursor;
    if (mapped.entity) {
      entities.push(mapped.entity);
    }
  }
  entities.sort((a, b) => a.start - b.start);

  return {
    entities,
    redactedText: redactText(text, entities),
  };
};
