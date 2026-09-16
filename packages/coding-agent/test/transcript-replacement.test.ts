/**
 * Durable transcript replacement in the coding-agent session layer.
 *
 * `SessionManager.appendTranscript()` writes a boundary entry: context construction
 * stops there and uses the entry's own messages, while everything earlier stays in the
 * session for history, forking and tree navigation.
 */

import type { AssistantMessage, UserMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { prepareCompaction } from "../src/core/compaction/index.ts";
import {
	buildContextEntries,
	buildSessionContext,
	SessionManager,
	type TranscriptEntry,
} from "../src/core/session-manager.ts";

function userMessage(text: string): UserMessage {
	return { role: "user", content: text, timestamp: 1 };
}

function assistantMessage(text: string): AssistantMessage {
	return {
		role: "assistant",
		content: [{ type: "text", text }],
		api: "anthropic-messages",
		provider: "anthropic",
		model: "test",
		usage: {
			input: 10,
			output: 5,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 15,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason: "stop",
		timestamp: 2,
	};
}

describe("SessionManager.appendTranscript", () => {
	it("writes a transcript entry that carries its messages and metadata", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));

		const id = session.appendTranscript([userMessage("state only")], {
			reason: "skill-state step 3",
			source: "skill-state",
			details: { step: 3 },
		});

		const entry = session.getEntry(id) as TranscriptEntry;
		expect(entry.type).toBe("transcript");
		expect(entry.messages).toEqual([userMessage("state only")]);
		expect(entry.reason).toBe("skill-state step 3");
		expect(entry.source).toBe("skill-state");
		expect(entry.details).toEqual({ step: 3 });
		expect(session.getLeafId()).toBe(id);
	});

	it("keeps the replaced history in the session but out of context", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));
		session.appendMessage(assistantMessage("a long reasoning trace"));
		session.appendTranscript([userMessage("state only")]);

		// History is intact for the tree, forking and the TUI...
		expect(session.getEntries()).toHaveLength(3);
		expect(session.getBranch()).toHaveLength(3);

		// ...but the LLM only sees the replacement.
		expect(buildSessionContext(session.getEntries(), session.getLeafId()).messages).toEqual([
			userMessage("state only"),
		]);
	});

	it("includes messages appended after the boundary", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));
		session.appendTranscript([userMessage("state only")]);
		session.appendMessage(userMessage("latest observation"));

		expect(buildSessionContext(session.getEntries(), session.getLeafId()).messages).toEqual([
			userMessage("state only"),
			userMessage("latest observation"),
		]);
	});

	it("keeps only the newest boundary", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));
		session.appendTranscript([userMessage("state v1")]);
		session.appendMessage(userMessage("middle"));
		session.appendTranscript([userMessage("state v2")]);

		expect(buildSessionContext(session.getEntries(), session.getLeafId()).messages).toEqual([
			userMessage("state v2"),
		]);
	});

	it("supersedes an earlier compaction", () => {
		const session = SessionManager.inMemory();
		const firstId = session.appendMessage(userMessage("original task"));
		session.appendCompaction("earlier summary", firstId, 1000);
		session.appendTranscript([userMessage("state only")]);

		const context = buildSessionContext(session.getEntries(), session.getLeafId());
		expect(context.messages).toEqual([userMessage("state only")]);
		expect(context.messages.some((message) => message.role === "compactionSummary")).toBe(false);
	});

	it("heads the context entry list so nothing earlier is walked", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));
		session.appendMessage(assistantMessage("reasoning"));
		const boundaryId = session.appendTranscript([userMessage("state only")]);
		session.appendMessage(userMessage("after"));

		const contextEntries = buildContextEntries(session.getEntries(), session.getLeafId());
		expect(contextEntries[0]?.id).toBe(boundaryId);
		expect(contextEntries).toHaveLength(2);
	});

	it("survives a round trip through session serialization", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original task"));
		session.appendTranscript([userMessage("state only")], { reason: "bounded" });

		// Entries are persisted as plain JSON records; reloading must preserve the boundary.
		const reloaded = JSON.parse(JSON.stringify(session.getEntries())) as ReturnType<typeof session.getEntries>;
		const context = buildSessionContext(reloaded, session.getLeafId());

		expect(context.messages).toEqual([userMessage("state only")]);
		expect((reloaded.find((entry) => entry.type === "transcript") as TranscriptEntry).reason).toBe("bounded");
	});
});

describe("compaction after a transcript replacement", () => {
	const settings = { enabled: true, reserveTokens: 1000, keepRecentTokens: 1 };

	it("never summarizes history from before the boundary", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("DISCARDED-HISTORY"));
		session.appendMessage(assistantMessage("DISCARDED-REASONING"));
		session.appendTranscript([userMessage("state only")]);
		session.appendMessage(userMessage("after boundary"));
		session.appendMessage(assistantMessage("answer after boundary"));

		const preparation = prepareCompaction(session.getBranch(), settings);

		expect(preparation).toBeDefined();
		const summarized = JSON.stringify([...preparation!.messagesToSummarize, ...preparation!.turnPrefixMessages]);
		expect(summarized).not.toContain("DISCARDED-HISTORY");
		expect(summarized).not.toContain("DISCARDED-REASONING");
	});

	it("carries every message of the replacement into the summary", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("DISCARDED-HISTORY"));
		session.appendTranscript([userMessage("state part one"), userMessage("state part two")]);
		session.appendMessage(userMessage("after boundary"));
		session.appendMessage(assistantMessage("answer after boundary"));

		const preparation = prepareCompaction(session.getBranch(), settings);

		const summarized = JSON.stringify([...preparation!.messagesToSummarize, ...preparation!.turnPrefixMessages]);
		expect(summarized).toContain("state part one");
		expect(summarized).toContain("state part two");
	});

	it("does not prepare a compaction when the branch ends at a replacement", () => {
		const session = SessionManager.inMemory();
		session.appendMessage(userMessage("original"));
		session.appendTranscript([userMessage("state only")]);

		expect(prepareCompaction(session.getBranch(), settings)).toBeUndefined();
	});
});

describe("stored replacement isolation", () => {
	it("does not alias the caller's array", () => {
		const session = SessionManager.inMemory();
		const live: (UserMessage | AssistantMessage)[] = [userMessage("state only")];
		session.appendTranscript(live);

		live.push(assistantMessage("answer"));

		expect(buildSessionContext(session.getEntries(), session.getLeafId()).messages).toEqual([
			userMessage("state only"),
		]);
	});
});
