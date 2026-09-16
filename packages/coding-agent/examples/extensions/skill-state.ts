/**
 * SKILL.state - bounded-context agent runtime
 *
 * Implements the runtime described in https://arxiv.org/html/2608.26263. Instead of
 * accumulating a conversation transcript, the agent keeps an explicit structured
 * execution state. Each step the model sees only the immutable skill specification,
 * the current state, and whatever has happened since the last step; its intermediate
 * reasoning is discarded once the state update is recorded. Prompt size is O(1) in
 * the number of steps rather than O(T).
 *
 * The mechanism is a single call. After the model records a state update, the
 * extension writes a transcript boundary:
 *
 *   pi.replaceTranscript([specAndState], { deliverAs: "steer" })
 *
 * Context construction stops at that boundary, so the next step's context is the
 * spec and state plus only the messages that arrive afterwards. Earlier entries stay
 * in the session for history, forking and the TUI, but are never sent to the model.
 *
 * Because the boundary is durable, this needs no `context` handler: the replacement
 * is not re-derived on every LLM call, it survives reload and branch navigation, and
 * pi's own context accounting sees the smaller transcript. Tool results keep their
 * matching tool call, since a boundary only ever lands at a turn boundary.
 *
 * Usage:
 *   /skill-state on <spec>   begin a run with the given skill specification
 *   /skill-state off         return to normal transcript-based operation
 *   /skill-state             show the current state
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

/** Custom entry type used to persist state across reloads and branch navigation. */
const STATE_ENTRY = "skill-state";

/** Name of the tool the model calls to record its state update. */
const SET_STATE_TOOL = "set_state";

/**
 * The durable state of a SKILL.state run.
 *
 * This is the sole representation of task progress - everything the agent knows about
 * what it has done lives here, not in the transcript.
 */
interface SkillState {
	/** Immutable skill specification for this run. */
	spec: string;
	/** Structured execution state, replaced wholesale on each step. */
	state: Record<string, unknown>;
	/** Number of state updates recorded so far. */
	step: number;
	/** Set once the model reports the task complete. */
	done: boolean;
}

const SetStateParams = Type.Object({
	// A free-form object rather than Type.Record: Type.Record emits `patternProperties`,
	// which Gemini and OpenAI strict function calling both reject.
	state: Type.Object(
		{},
		{
			additionalProperties: true,
			description:
				"The complete updated execution state. This REPLACES the previous state entirely, so carry forward every field you still need - anything you omit is forgotten permanently.",
		},
	),
	done: Type.Optional(
		Type.Boolean({
			description: "Set to true when the specification has been fully satisfied.",
		}),
	),
});

interface SetStateDetails {
	state: Record<string, unknown>;
	step: number;
	done: boolean;
}

/** The single message that stands in for the entire conversation so far. */
function specAndState(current: SkillState): string {
	return [
		"# Skill specification",
		current.spec,
		"",
		"# Current state",
		"```json",
		JSON.stringify(current.state, null, 2),
		"```",
		"",
		`# Step ${current.step + 1}`,
		"",
		`You cannot see earlier steps - the state above is all that carried over. Act on the specification, then call \`${SET_STATE_TOOL}\` with the complete updated state before you finish; any field you omit is lost.`,
	].join("\n");
}

export default function skillStateExtension(pi: ExtensionAPI) {
	let current: SkillState | null = null;

	/** Rebuild state from the current branch. Later entries win. */
	const reconstructState = (ctx: ExtensionContext) => {
		current = null;
		for (const entry of ctx.sessionManager.getBranch()) {
			if (entry.type !== "custom" || entry.customType !== STATE_ENTRY) continue;
			const data = entry.data as SkillState | undefined;
			if (data) current = data;
		}
	};

	const persist = () => {
		if (current) pi.appendEntry<SkillState>(STATE_ENTRY, current);
	};

	/**
	 * Collapse the transcript down to the spec and current state.
	 *
	 * Queued as a steer so it lands at the end of the current turn rather than between
	 * the model's tool call and its result.
	 */
	const bound = (reason: string) => {
		if (!current) return;
		pi.replaceTranscript([{ role: "user", content: specAndState(current), timestamp: Date.now() }], {
			reason,
			source: "skill-state",
			details: { step: current.step },
			deliverAs: "steer",
		});
	};

	pi.on("session_start", async (_event, ctx) => reconstructState(ctx));
	pi.on("session_tree", async (_event, ctx) => reconstructState(ctx));

	pi.registerTool({
		name: SET_STATE_TOOL,
		label: "Set state",
		description:
			"Record the updated execution state. The state you pass REPLACES the previous state entirely and is the only thing that survives to the next step - your reasoning and this conversation are discarded.",
		parameters: SetStateParams,
		execute: async (_toolCallId, params) => {
			if (!current) {
				throw new Error("No SKILL.state run is active. Start one with /skill-state on <spec>.");
			}
			if (current.done) {
				throw new Error("This SKILL.state run has finished. Start a new one with /skill-state on <spec>.");
			}

			current = {
				...current,
				state: params.state as Record<string, unknown>,
				step: current.step + 1,
				done: params.done ?? false,
			};
			persist();
			// A finished run keeps its final transcript rather than collapsing again.
			if (!current.done) bound(`skill-state step ${current.step}`);

			const details: SetStateDetails = { state: current.state, step: current.step, done: current.done };
			return {
				content: [{ type: "text", text: `State recorded (step ${current.step}).` }],
				details,
				...(current.done ? { terminate: true } : {}),
			};
		},
	});

	pi.registerCommand("skill-state", {
		description: "Run the agent on bounded structured state instead of a transcript",
		getArgumentCompletions: (prefix) => {
			const options = ["on", "off"].filter((option) => option.startsWith(prefix));
			return options.length > 0 ? options.map((option) => ({ value: option, label: option })) : null;
		},
		handler: async (args, ctx) => {
			const trimmed = args.trim();

			if (trimmed === "off") {
				if (!current) {
					ctx.ui.notify("No SKILL.state run is active.", "info");
					return;
				}
				current = { ...current, done: true };
				persist();
				ctx.ui.notify("SKILL.state disabled - the transcript accumulates normally again.", "info");
				return;
			}

			if (trimmed.startsWith("on")) {
				const spec = trimmed.slice(2).trim();
				if (spec.length === 0) {
					ctx.ui.notify("Provide a specification: /skill-state on <spec>", "warning");
					return;
				}
				current = { spec, state: {}, step: 0, done: false };
				persist();
				bound("skill-state run started");
				ctx.ui.notify("SKILL.state enabled. Context is now spec + state + what follows.", "info");
				return;
			}

			if (!current) {
				ctx.ui.notify("No SKILL.state run is active. Start one with /skill-state on <spec>.", "info");
				return;
			}

			const status = current.done ? "done" : `step ${current.step}`;
			ctx.ui.notify(`SKILL.state (${status})\n${JSON.stringify(current.state, null, 2)}`, "info");
		},
	});
}
