/**
 * Delivery timing for AgentSession.replaceTranscript().
 *
 * Replacing context between a tool call and its result would produce a message array
 * providers reject, so a replacement requested mid-run is queued until a safe point.
 * These tests pin which point each `deliverAs` mode lands on.
 */

import { existsSync, mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent } from "@earendil-works/pi-agent-core";
import { createAssistantMessageEventStream, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { getModel, streamSimple } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AgentSession } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestResourceLoader } from "./utilities.ts";

describe("AgentSession.replaceTranscript delivery", () => {
	let session: AgentSession;
	let sessionManager: SessionManager;
	let tempDir: string;

	beforeEach(async () => {
		tempDir = join(tmpdir(), `pi-transcript-delivery-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			streamFn: streamSimple,
			initialState: { model, systemPrompt: "Test", tools: [] },
		});

		sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(),
		});
		session.subscribe(() => {});

		session.agent.streamFunction = (streamModel) => {
			const stream = createAssistantMessageEventStream();
			void Promise.resolve().then(() => {
				stream.push({
					type: "done",
					reason: "stop",
					message: {
						...fauxAssistantMessage("ok"),
						api: streamModel.api,
						provider: streamModel.provider,
						model: streamModel.id,
					},
				});
			});
			return stream;
		};
	});

	afterEach(() => {
		session.dispose();
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	const replacement = [{ role: "user" as const, content: "state only", timestamp: 1 }];

	const transcriptEntries = () => sessionManager.getEntries().filter((entry) => entry.type === "transcript");

	it("applies immediately when the agent is idle", async () => {
		sessionManager.appendMessage({ role: "user", content: "original", timestamp: 1 });

		await session.replaceTranscript(replacement, { reason: "idle write" });

		expect(transcriptEntries()).toHaveLength(1);
		expect(session.agent.state.messages).toEqual(replacement);
		expect(sessionManager.buildSessionContext().messages).toEqual(replacement);
	});

	it("defers a nextTurn replacement even when idle", async () => {
		sessionManager.appendMessage({ role: "user", content: "original", timestamp: 1 });

		await session.replaceTranscript(replacement, { deliverAs: "nextTurn" });

		// Nothing written yet - it waits for the next prompt.
		expect(transcriptEntries()).toHaveLength(0);
	});

	it("queues a steer replacement while streaming and applies it at the turn boundary", async () => {
		// Gate the response so the request is provably mid-run when the replacement lands.
		let releaseResponse!: () => void;
		const responseGate = new Promise<void>((resolve) => {
			releaseResponse = resolve;
		});
		session.agent.streamFunction = (streamModel) => {
			const stream = createAssistantMessageEventStream();
			void responseGate.then(() => {
				stream.push({
					type: "done",
					reason: "stop",
					message: {
						...fauxAssistantMessage("ok"),
						api: streamModel.api,
						provider: streamModel.provider,
						model: streamModel.id,
					},
				});
			});
			return stream;
		};

		const promptPromise = session.prompt("hello");
		while (!session.isStreaming) await new Promise((resolve) => setImmediate(resolve));

		await session.replaceTranscript(replacement, { deliverAs: "steer" });
		// Still queued: rewriting context mid-turn would strand the in-flight response.
		expect(transcriptEntries()).toHaveLength(0);

		releaseResponse();
		await promptPromise;

		// The boundary is the last entry, so it wins over everything the turn produced.
		expect(transcriptEntries()).toHaveLength(1);
		expect(sessionManager.buildSessionContext().messages).toEqual(replacement);
	});

	it("records the boundary metadata it was given", async () => {
		await session.replaceTranscript(replacement, { reason: "step 7", source: "skill-state", details: { step: 7 } });

		const entry = transcriptEntries()[0];
		expect(entry).toMatchObject({ reason: "step 7", source: "skill-state", details: { step: 7 } });
	});

	it("drops the replaced history from context but keeps it in the session", async () => {
		sessionManager.appendMessage({ role: "user", content: "original", timestamp: 1 });
		sessionManager.appendMessage({ role: "user", content: "more history", timestamp: 2 });

		await session.replaceTranscript(replacement);

		expect(sessionManager.getEntries().length).toBeGreaterThan(1);
		expect(sessionManager.buildSessionContext().messages).toEqual(replacement);
	});
});

describe("AgentSession.replaceTranscript reaches the provider", () => {
	let session: AgentSession;
	let sessionManager: SessionManager;
	let tempDir: string;

	const REPLACEMENT = "STATE-ONLY-MARKER";
	const DISCARDED = "DISCARDED-HISTORY-MARKER";

	beforeEach(async () => {
		tempDir = join(tmpdir(), `pi-transcript-provider-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });

		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			streamFn: streamSimple,
			initialState: { model, systemPrompt: "Test", tools: [] },
		});

		sessionManager = SessionManager.inMemory();
		const settingsManager = SettingsManager.create(tempDir, tempDir);
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);

		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager,
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(),
			customTools: [
				{
					name: "bound",
					label: "Bound",
					description: "Replace the transcript",
					parameters: Type.Object({}),
					execute: async () => {
						await session.replaceTranscript([{ role: "user", content: REPLACEMENT, timestamp: 1 }], {
							deliverAs: "steer",
						});
						return { content: [{ type: "text", text: "bounded" }], details: {} };
					},
				},
			],
		});
		session.subscribe(() => {});
	});

	afterEach(() => {
		session.dispose();
		vi.restoreAllMocks();
		if (tempDir && existsSync(tempDir)) rmSync(tempDir, { recursive: true });
	});

	it("sends the replacement, not the discarded history, on the next provider request", async () => {
		const requests: string[][] = [];
		let call = 0;
		session.agent.streamFunction = (streamModel, context) => {
			requests.push(context.messages.map((message) => JSON.stringify(message)));
			const stream = createAssistantMessageEventStream();
			const first = call++ === 0;
			void Promise.resolve().then(() => {
				stream.push({
					type: "done",
					reason: first ? "toolUse" : "stop",
					message: {
						...fauxAssistantMessage(first ? "" : "done"),
						...(first ? { content: [{ type: "toolCall", id: "call-1", name: "bound", arguments: {} }] } : {}),
						stopReason: first ? "toolUse" : "stop",
						api: streamModel.api,
						provider: streamModel.provider,
						model: streamModel.id,
					},
				});
			});
			return stream;
		};

		await session.prompt(DISCARDED);

		expect(requests.length).toBeGreaterThanOrEqual(2);
		const second = requests[1].join("\n");
		expect(second).toContain(REPLACEMENT);
		expect(second).not.toContain(DISCARDED);
	});
});
