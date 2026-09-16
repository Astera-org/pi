/**
 * Context accounting for extensions that rewrite context via the `context` event.
 *
 * An extension can replace the message array pi sends to the provider (see
 * `examples/extensions/skill-state.ts`, which collapses an O(T) transcript into an
 * O(1) prompt). That rewrite happens in `transformContext`, at the very end of the
 * agent loop - pi's own session keeps every message.
 *
 * These tests pin how pi's compaction accounting behaves in that situation, because
 * a naive reading ("pi measures the raw transcript, so it will compact a history the
 * model never sees") is wrong in the common case and right in one narrow case.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { AssistantMessage, Usage } from "@earendil-works/pi-ai/compat";
import { describe, expect, it } from "vitest";
import { estimateContextTokens } from "../src/core/compaction/index.ts";

function usage(input: number, output: number): Usage {
	return {
		input,
		output,
		cacheRead: 0,
		cacheWrite: 0,
		totalTokens: input + output,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

function assistant(text: string, overrides: Partial<AssistantMessage> = {}): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		usage: usage(1_200, 200),
		stopReason: "stop",
		timestamp: Date.now(),
		api: "anthropic-messages",
		provider: "anthropic",
		model: "claude-sonnet-4-5",
		...overrides,
	};
}

/**
 * A transcript shaped like a long SKILL.state run: many accumulated steps, but every
 * assistant response was produced from a small rewritten prompt, so its usage is small.
 */
function longRewrittenRun(steps: number): AgentMessage[] {
	const bulky = "x".repeat(4_000);
	const messages: AgentMessage[] = [];
	for (let step = 0; step < steps; step++) {
		messages.push(assistant(`step ${step}`));
		messages.push({
			role: "toolResult",
			toolCallId: `call-${step}`,
			toolName: "read",
			content: [{ type: "text", text: bulky }],
			isError: false,
		} as AgentMessage);
	}
	return messages;
}

describe("context accounting with a context-rewriting extension", () => {
	it("stays bounded in steady state, because the estimate anchors on real provider usage", () => {
		const messages = longRewrittenRun(60);

		const estimate = estimateContextTokens(messages);

		// The raw transcript is ~240k characters of tool output. If the estimate summed
		// the raw array it would be tens of thousands of tokens.
		const rawCharacters = 60 * 4_000;
		expect(rawCharacters).toBeGreaterThan(200_000);

		// Instead it anchors on the last assistant's actual usage (1,400) plus only the
		// messages after it, so a rewriting extension does not trip compaction.
		expect(estimate.usageTokens).toBe(1_400);
		expect(estimate.tokens).toBeLessThan(5_000);
	});

	it("falls back to the previous successful usage when the last response errored", () => {
		const messages = longRewrittenRun(60);
		messages.push(assistant("boom", { stopReason: "error", usage: usage(0, 0) }));

		const estimate = estimateContextTokens(messages);

		// Zero-usage and errored assistants are skipped, so the anchor is still a real
		// post-rewrite measurement rather than the raw history.
		expect(estimate.usageTokens).toBe(1_400);
		expect(estimate.tokens).toBeLessThan(5_000);
	});

	it("measures the raw transcript only when no assistant carries usable usage", () => {
		// This is the one case where pi's estimate reflects a history the model was
		// never sent: every assistant response failed, so there is no usage to anchor on.
		const messages = longRewrittenRun(60).map((message) =>
			message.role === "assistant" ? assistant("failed", { stopReason: "error", usage: usage(0, 0) }) : message,
		);

		const estimate = estimateContextTokens(messages);

		expect(estimate.lastUsageIndex).toBeNull();
		expect(estimate.usageTokens).toBe(0);
		expect(estimate.tokens).toBeGreaterThan(20_000);
	});
});
