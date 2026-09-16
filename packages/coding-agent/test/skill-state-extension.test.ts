/**
 * Tests for the SKILL.state example extension (examples/extensions/skill-state.ts).
 *
 * The claim being validated: with pi.replaceTranscript(), an extension can hold the
 * model-visible context at O(1) while the session keeps full history - and it does so
 * durably, without re-deriving a projection on every LLM call.
 */

import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { describe, expect, it, vi } from "vitest";
import skillStateExtension from "../examples/extensions/skill-state.ts";
import type { ExtensionAPI, ExtensionContext } from "../src/core/extensions/index.ts";

type CommandHandler = (args: string, ctx: ExtensionContext) => Promise<void> | void;
type SessionHandler = (event: unknown, ctx: ExtensionContext) => Promise<void> | void;

interface ReplaceCall {
	messages: AgentMessage[];
	options?: { reason?: string; source?: string; details?: unknown; deliverAs?: string };
}

interface SetStateParams {
	state: Record<string, unknown>;
	done?: boolean;
}

interface RegisteredTool {
	name: string;
	execute: (toolCallId: string, params: SetStateParams) => Promise<unknown>;
}

interface CustomEntryRecord {
	type: "custom";
	customType: string;
	data: unknown;
}

function setup() {
	const commands = new Map<string, CommandHandler>();
	const handlers = new Map<string, SessionHandler>();
	const entries: CustomEntryRecord[] = [];
	const notifications: string[] = [];
	const replacements: ReplaceCall[] = [];
	let tool: RegisteredTool | undefined;

	const api = {
		registerFlag: vi.fn(),
		registerShortcut: vi.fn(),
		registerCommand(name: string, command: { handler: CommandHandler }) {
			commands.set(name, command.handler);
		},
		registerTool(definition: RegisteredTool) {
			tool = definition;
		},
		on(event: string, handler: unknown) {
			handlers.set(event, handler as SessionHandler);
		},
		appendEntry(customType: string, data: unknown) {
			entries.push({ type: "custom", customType, data });
		},
		replaceTranscript(messages: AgentMessage[], options?: ReplaceCall["options"]) {
			replacements.push({ messages, ...(options === undefined ? {} : { options }) });
		},
	} as unknown as ExtensionAPI;

	const ctx = {
		ui: { notify: (message: string) => notifications.push(message) },
		sessionManager: { getBranch: () => entries },
	} as unknown as ExtensionContext;

	skillStateExtension(api);

	return {
		ctx,
		entries,
		notifications,
		replacements,
		runCommand: (args: string) => commands.get("skill-state")!(args, ctx),
		sessionStart: async () => {
			await handlers.get("session_start")!({ type: "session_start" }, ctx);
		},
		setState: (params: SetStateParams) => tool!.execute("call-1", params),
	};
}

/** Text of the single user message a replacement carries. */
function replacementText(call: ReplaceCall): string {
	expect(call.messages).toHaveLength(1);
	const message = call.messages[0];
	if (message.role !== "user") throw new Error(`expected a user message, got ${message.role}`);
	return typeof message.content === "string"
		? message.content
		: message.content
				.filter((part) => part.type === "text")
				.map((part) => (part as { text: string }).text)
				.join("");
}

describe("skill-state extension", () => {
	it("does not touch the transcript until a run is started", () => {
		const harness = setup();
		expect(harness.replacements).toHaveLength(0);
	});

	it("bounds the transcript when a run starts", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");

		expect(harness.replacements).toHaveLength(1);
		expect(replacementText(harness.replacements[0])).toContain("build a parser");
	});

	it("queues replacements as steers so they land at a turn boundary", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		await harness.setState({ state: { phase: "lexing" } });

		for (const call of harness.replacements) {
			expect(call.options?.deliverAs).toBe("steer");
		}
	});

	it("re-bounds after every state update, carrying spec and state only", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		await harness.setState({ state: { phase: "lexing", tokensDone: 12 } });

		const latest = replacementText(harness.replacements[harness.replacements.length - 1]);
		expect(latest).toContain("build a parser");
		expect(latest).toContain("lexing");
		expect(latest).toContain("tokensDone");
		expect(harness.replacements).toHaveLength(2);
	});

	it("keeps the prompt bounded as steps accumulate", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");

		// A fixed-size state at step 1 and at step 40 produces the same sized context.
		await harness.setState({ state: { phase: "a" } });
		const early = replacementText(harness.replacements[harness.replacements.length - 1]);
		for (let step = 0; step < 40; step++) await harness.setState({ state: { phase: "a" } });
		const late = replacementText(harness.replacements[harness.replacements.length - 1]);

		// Only the step counter grows, and only by its digit count.
		expect(late.length - early.length).toBeLessThanOrEqual(2);
	});

	it("tags boundaries so they are identifiable in the session", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		await harness.setState({ state: { phase: "lexing" } });

		const call = harness.replacements[harness.replacements.length - 1];
		expect(call.options?.source).toBe("skill-state");
		expect(call.options?.reason).toContain("step 1");
		expect(call.options?.details).toEqual({ step: 1 });
	});

	it("persists state and restores it on session start", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		await harness.setState({ state: { phase: "parsing" } });

		// Reload: a fresh extension instance reading the same branch entries.
		const reloaded = setup();
		reloaded.entries.push(...harness.entries);
		await reloaded.sessionStart();
		await reloaded.setState({ state: { phase: "emitting" } });

		const latest = replacementText(reloaded.replacements[reloaded.replacements.length - 1]);
		expect(latest).toContain("build a parser");
		expect(latest).toContain("emitting");
	});

	it("stops bounding once the run is done", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		const before = harness.replacements.length;

		await harness.setState({ state: { phase: "complete" }, done: true });

		expect(harness.replacements).toHaveLength(before);
	});

	it("stops bounding when switched off, and will not silently re-enable", async () => {
		const harness = setup();
		await harness.runCommand("on build a parser");
		await harness.runCommand("off");
		const after = harness.replacements.length;

		// A stray tool call must not resurrect the run and start rewriting context again.
		await expect(harness.setState({ state: {} })).rejects.toThrow(/finished/);
		expect(harness.replacements).toHaveLength(after);
	});
});
