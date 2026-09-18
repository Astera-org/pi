import { beforeEach, describe, expect, it, vi } from "vitest";
import { stream as streamOpenAICompletions } from "../src/api/openai-completions.ts";
import type { Model } from "../src/types.ts";
import { normalizeContext } from "../src/utils/transcript.ts";

const mockState = vi.hoisted(() => ({
	chunks: [] as unknown[],
	lastParams: undefined as unknown,
}));

vi.mock("openai", () => {
	class FakeOpenAI {
		chat = {
			completions: {
				create: (params: unknown) => {
					mockState.lastParams = params;
					const chunks = mockState.chunks;
					const stream = {
						async *[Symbol.asyncIterator]() {
							for (const chunk of chunks) yield chunk;
						},
					};
					const promise = Promise.resolve(stream) as Promise<typeof stream> & {
						withResponse: () => Promise<{
							data: typeof stream;
							response: { status: number; headers: Headers };
						}>;
					};
					promise.withResponse = async () => ({
						data: stream,
						response: { status: 200, headers: new Headers() },
					});
					return promise;
				},
			},
		};
	}
	return { default: FakeOpenAI };
});

const model: Model<"openai-completions"> = {
	id: "test-model",
	name: "Test Model",
	api: "openai-completions",
	provider: "openai",
	baseUrl: "https://api.openai.com/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
};

const context = normalizeContext({
	messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
});

describe("OpenAI completions logprobs", () => {
	beforeEach(() => {
		mockState.chunks = [];
		mockState.lastParams = undefined;
	});

	it("sends logprobs: true when requested without topLogprobs", async () => {
		mockState.chunks = [{ id: "chatcmpl-1", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }];

		await streamOpenAICompletions(model, context, { apiKey: "test", logprobs: true }).result();

		const params = mockState.lastParams as { logprobs?: boolean; top_logprobs?: number };
		expect(params.logprobs).toBe(true);
		expect(params.top_logprobs).toBeUndefined();
	});

	it("sends logprobs: true and top_logprobs: N when both are requested", async () => {
		mockState.chunks = [{ id: "chatcmpl-2", choices: [{ index: 0, delta: {}, finish_reason: "stop" }] }];

		await streamOpenAICompletions(model, context, {
			apiKey: "test",
			logprobs: true,
			topLogprobs: 3,
		}).result();

		const params = mockState.lastParams as { logprobs?: boolean; top_logprobs?: number };
		expect(params.logprobs).toBe(true);
		expect(params.top_logprobs).toBe(3);
	});

	it("accumulates per-token logprobs onto the text content block", async () => {
		mockState.chunks = [
			{
				id: "chatcmpl-3",
				choices: [
					{
						index: 0,
						delta: { content: "Hel" },
						logprobs: {
							content: [
								{
									token: "Hel",
									logprob: -0.1,
									bytes: [72, 101, 108],
									top_logprobs: [{ token: "Hel", logprob: -0.1, bytes: [72, 101, 108] }],
								},
							],
						},
						finish_reason: null,
					},
				],
			},
			{
				id: "chatcmpl-3",
				choices: [
					{
						index: 0,
						delta: { content: "lo" },
						logprobs: {
							content: [{ token: "lo", logprob: -0.05, bytes: [108, 111], top_logprobs: [] }],
						},
						finish_reason: "stop",
					},
				],
			},
		];

		const message = await streamOpenAICompletions(model, context, {
			apiKey: "test",
			logprobs: true,
			topLogprobs: 1,
		}).result();

		const textBlock = message.content.find((block) => block.type === "text");
		expect(textBlock?.text).toBe("Hello");
		expect(textBlock?.logprobs).toEqual([
			{
				token: "Hel",
				logprob: -0.1,
				bytes: [72, 101, 108],
				topLogprobs: [{ token: "Hel", logprob: -0.1, bytes: [72, 101, 108] }],
			},
			{
				token: "lo",
				logprob: -0.05,
				bytes: [108, 111],
				topLogprobs: undefined,
			},
		]);
	});

	it("does not send or accumulate logprobs when not requested", async () => {
		mockState.chunks = [
			{ id: "chatcmpl-4", choices: [{ index: 0, delta: { content: "hi" }, finish_reason: "stop" }] },
		];

		const message = await streamOpenAICompletions(model, context, { apiKey: "test" }).result();

		const params = mockState.lastParams as { logprobs?: boolean; top_logprobs?: number };
		expect("logprobs" in (params as object)).toBe(false);
		expect("top_logprobs" in (params as object)).toBe(false);

		const textBlock = message.content.find((block) => block.type === "text");
		expect(textBlock?.logprobs).toBeUndefined();
	});
});
