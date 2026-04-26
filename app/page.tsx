"use client";

import { useMemo, useState } from "react";
import { toast } from "sonner";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { detectPII, ensurePIIPipeline, isPipelineReady, SUPPORTED_LABELS, type DetectionResult } from "@/lib/pii";

const SAMPLE_TEXT = `Hi team — please email me at rachel.owens@acme-payments.com or call +1 (415) 555-0179.
My backup number is 415-555-0198 and SSN is 123-45-6789.
You can also review docs at https://acme-payments.example/internal/claims before Friday.`;

type ViewState = "idle" | "model-loading" | "detecting" | "success" | "empty-result" | "error";
type BatchSample = { id: string; text: string };
type BatchRunResult = {
  id: string;
  text: string;
  ms: number;
  entityCount: number;
  foundPii: boolean;
};

const BENCHMARK_SAMPLES: BatchSample[] = [
  { id: "s1", text: "Email me at mia.chen@northwind.com for invoice update." },
  { id: "s2", text: "Daily standup at 10:30 with product and design teams." },
  { id: "s3", text: "Reach me on +1 (212) 555-0111 before shipping closes." },
  { id: "s4", text: "Incident happened in us-east-1, service restored now." },
  { id: "s5", text: "Client SSN: 123-45-6789. Store in secure vault only." },
  { id: "s6", text: "Roadmap draft shared in Notion. No customer data included." },
  { id: "s7", text: "User profile URL is https://portal.example.com/u/harry-91." },
  { id: "s8", text: "Next sprint scope: retry queue, dead-letter metrics, alerts." },
  { id: "s9", text: "Driver license noted as D123-4567-8901 in intake form." },
  { id: "s10", text: "Company offsite in Denver moved to next month." },
  { id: "s11", text: "Backup contact: 415-555-0198 and jane.doe@example.org." },
  { id: "s12", text: "Feature flag rollout at 25% then 50% then 100%." },
  { id: "s13", text: "Passport number recorded as X12345678 for visa paperwork." },
  { id: "s14", text: "Refactor parser to avoid quadratic merge in hot path." },
  { id: "s15", text: "Charge card 4111-1111-1111-1111 for sandbox billing test." },
  { id: "s16", text: "Public docs live at http://docs.contoso.dev/guide/setup." },
  { id: "s17", text: "Team lunch Friday; share dietary prefs in chat thread." },
  { id: "s18", text: "Employee ITIN: 912-34-5678 present in HR upload." },
  { id: "s19", text: "Latest build passed lint, typecheck, and e2e smoke tests." },
  { id: "s20", text: "Server login password temporarily set to Winter#2026." },
];

export default function Home() {
  const [input, setInput] = useState(SAMPLE_TEXT);
  const [status, setStatus] = useState<ViewState>("idle");
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [batchResults, setBatchResults] = useState<BatchRunResult[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchError, setBatchError] = useState<string | null>(null);

  const isBusy = status === "model-loading" || status === "detecting";
  const disableDetect = !input.trim() || isBusy;

  const statusLabel = useMemo(() => {
    if (status === "model-loading") {
      return "Downloading model from Hugging Face...";
    }
    if (status === "detecting") {
      return "Running local PII detection...";
    }
    return null;
  }, [status]);

  const onDetect = async () => {
    if (!input.trim()) {
      toast.error("Enter text first.");
      return;
    }

    setErrorMessage(null);
    setResult(null);

    try {
      if (!isPipelineReady()) {
        setStatus("model-loading");
        await ensurePIIPipeline();
      }

      setStatus("detecting");
      const detection = await detectPII(input);
      setResult(detection);
      setStatus(detection.entities.length ? "success" : "empty-result");
    } catch (error) {
      setStatus("error");
      const message = error instanceof Error ? error.message : "Unknown detection error.";
      setErrorMessage(message);
    }
  };

  const copyText = async (text: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast.success(successMessage);
    } catch {
      toast.error("Clipboard write failed.");
    }
  };

  const runBenchmark = async () => {
    setBatchError(null);
    setBatchResults([]);
    setBatchRunning(true);

    try {
      if (!isPipelineReady()) {
        setStatus("model-loading");
        await ensurePIIPipeline();
        setStatus("idle");
      }

      const output: BatchRunResult[] = [];
      for (const sample of BENCHMARK_SAMPLES) {
        const start = performance.now();
        const detection = await detectPII(sample.text);
        const end = performance.now();
        output.push({
          id: sample.id,
          text: sample.text,
          ms: Number((end - start).toFixed(1)),
          entityCount: detection.entities.length,
          foundPii: detection.entities.length > 0,
        });
      }
      setBatchResults(output);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Benchmark run failed.";
      setBatchError(message);
    } finally {
      setBatchRunning(false);
    }
  };

  const batchSummary = useMemo(() => {
    if (!batchResults.length) {
      return null;
    }
    const total = batchResults.reduce((sum, item) => sum + item.ms, 0);
    const avg = total / batchResults.length;
    const found = batchResults.filter((item) => item.foundPii).length;
    return {
      totalMs: Number(total.toFixed(1)),
      avgMs: Number(avg.toFixed(1)),
      found,
      notFound: batchResults.length - found,
    };
  }, [batchResults]);

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-6 p-4 md:p-8">
      <Card>
        <CardHeader>
          <CardTitle>Browser-only PII Detector Demo</CardTitle>
          <CardDescription>
            Inference runs on-device in your browser using Transformers.js + ONNX. Input text is not sent to your server.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <Alert>
            <AlertTitle>Model caveats</AlertTitle>
            <AlertDescription>
              English-first model. False positives and false negatives happen. Only network call in this flow is model asset fetch from Hugging Face CDN.
            </AlertDescription>
          </Alert>

          <div className="space-y-2">
            <p className="text-sm font-medium">Input text</p>
            <Textarea
              className="min-h-44"
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="Paste text with possible PII..."
            />
          </div>

          <div className="flex flex-wrap gap-2">
            <Button onClick={onDetect} disabled={disableDetect}>
              Detect PII
            </Button>
            <Button variant="secondary" onClick={runBenchmark} disabled={isBusy || batchRunning}>
              {batchRunning ? "Running 20 samples..." : "Run 20 samples"}
            </Button>
            <Button variant="outline" onClick={() => setInput(SAMPLE_TEXT)} disabled={isBusy}>
              Reset sample
            </Button>
            <Button variant="outline" onClick={() => setInput("")} disabled={isBusy}>
              Clear
            </Button>
          </div>

          {statusLabel ? <p className="text-sm text-muted-foreground">{statusLabel}</p> : null}
          {status === "error" && errorMessage ? (
            <Alert variant="destructive">
              <AlertTitle>Detection failed</AlertTitle>
              <AlertDescription>{errorMessage}</AlertDescription>
            </Alert>
          ) : null}
          {batchError ? (
            <Alert variant="destructive">
              <AlertTitle>Benchmark failed</AlertTitle>
              <AlertDescription>{batchError}</AlertDescription>
            </Alert>
          ) : null}
        </CardContent>
      </Card>

      <div className="grid gap-6 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Detected entities</CardTitle>
            <CardDescription>Sorted by source text order.</CardDescription>
          </CardHeader>
          <CardContent>
            {isBusy ? (
              <div className="space-y-2">
                <Skeleton className="h-6 w-full" />
                <Skeleton className="h-6 w-[92%]" />
                <Skeleton className="h-6 w-[80%]" />
              </div>
            ) : null}

            {!isBusy && status === "idle" ? (
              <p className="text-sm text-muted-foreground">Run detection to view entities.</p>
            ) : null}

            {!isBusy && status === "empty-result" ? (
              <p className="text-sm text-muted-foreground">No PII detected in current input.</p>
            ) : null}

            {!isBusy && result?.entities.length ? (
              <ScrollArea className="h-64 pr-4">
                <div className="space-y-2">
                  {result.entities.map((entity, index) => (
                    <div key={`${entity.start}-${entity.end}-${index}`} className="rounded-md border p-3 text-sm">
                      <div className="mb-1 flex flex-wrap items-center gap-2">
                        <Badge>{entity.label}</Badge>
                        <span className="text-muted-foreground">
                          {entity.start}-{entity.end}
                        </span>
                        <span className="text-muted-foreground">score: {entity.score.toFixed(3)}</span>
                      </div>
                      <p className="break-all font-mono text-xs">{entity.text}</p>
                    </div>
                  ))}
                </div>
              </ScrollArea>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Redacted output</CardTitle>
            <CardDescription>Non-PII text preserved. Detected spans replaced with placeholders.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {isBusy ? <Skeleton className="h-40 w-full" /> : null}
            {!isBusy ? (
              <Textarea
                className="min-h-40 font-mono text-sm"
                readOnly
                value={result?.redactedText ?? ""}
                placeholder="Redacted text appears here after detection."
              />
            ) : null}
            <Button
              variant="outline"
              disabled={!result?.redactedText || isBusy}
              onClick={() => copyText(result?.redactedText ?? "", "Redacted text copied.")}
            >
              Copy redacted output
            </Button>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Batch benchmark (20 samples)</CardTitle>
          <CardDescription>Per-sample timing + PII/no-PII status colors.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {batchRunning ? (
            <div className="space-y-2">
              <Skeleton className="h-6 w-full" />
              <Skeleton className="h-6 w-[95%]" />
              <Skeleton className="h-6 w-[90%]" />
              <Skeleton className="h-6 w-[85%]" />
            </div>
          ) : null}

          {!batchRunning && !batchResults.length ? (
            <p className="text-sm text-muted-foreground">Click &quot;Run 20 samples&quot; to execute benchmark.</p>
          ) : null}

          {batchSummary ? (
            <div className="flex flex-wrap gap-2">
              <Badge variant="secondary">Total: {batchSummary.totalMs} ms</Badge>
              <Badge variant="secondary">Avg: {batchSummary.avgMs} ms</Badge>
              <Badge className="bg-emerald-600 text-white hover:bg-emerald-600">
                PII found: {batchSummary.found}
              </Badge>
              <Badge className="bg-red-600 text-white hover:bg-red-600">
                No PII: {batchSummary.notFound}
              </Badge>
            </div>
          ) : null}

          {!batchRunning && batchResults.length ? (
            <ScrollArea className="h-80 pr-4">
              <div className="space-y-2">
                {batchResults.map((item) => (
                  <div key={item.id} className="rounded-md border p-3 text-sm">
                    <div className="mb-2 flex flex-wrap items-center gap-2">
                      <Badge variant="outline">{item.id}</Badge>
                      <Badge variant="secondary">{item.ms} ms</Badge>
                      <Badge
                        className={
                          item.foundPii
                            ? "bg-emerald-600 text-white hover:bg-emerald-600"
                            : "bg-red-600 text-white hover:bg-red-600"
                        }
                      >
                        {item.foundPii ? `PII found (${item.entityCount})` : "No PII found"}
                      </Badge>
                    </div>
                    <p className="text-muted-foreground">{item.text}</p>
                  </div>
                ))}
              </div>
            </ScrollArea>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Supported labels</CardTitle>
          <CardDescription>From model card. Mappings used for redaction placeholders.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap gap-2">
            {SUPPORTED_LABELS.map((label) => (
              <Badge key={label} variant="secondary">
                {label}
              </Badge>
            ))}
          </div>
          <Separator />
          <Button
            variant="outline"
            disabled={!result?.entities.length || isBusy}
            onClick={() =>
              copyText(
                JSON.stringify(result?.entities ?? [], null, 2),
                "Entity list copied as JSON."
              )
            }
          >
            Copy entity JSON
          </Button>
        </CardContent>
      </Card>
    </main>
  );
}
