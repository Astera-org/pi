# FORK.md

`Astera-org/pi` is a fork of [`earendil-works/pi`](https://github.com/earendil-works/pi). It carries a small number of capabilities upstream doesn't have, and periodically merges upstream to stay current.

## Why this fork exists

This fork adds two capabilities that don't exist in the upstream `@earendil-works/pi-coding-agent` extension API:

- **`pi.replaceTranscript(messages, options?)`** (`packages/coding-agent/src/core/extensions/types.ts`): lets an extension write a durable boundary into the session's transcript mid-turn — context construction stops at that point and uses the replacement `messages` on every future request, until a later boundary supersedes it. Unlike the `context` event, which only reshapes a single request, this is recorded in the session and survives reload and branch navigation. It's used by `Astera-org/pi-state` to bound a prompt with a durable summary instead of full history.
- **Token-level logprobs in the `openai-completions` adapter** (`packages/ai/src/api/openai-completions.ts`, `packages/ai/src/types.ts`): `StreamOptions.logprobs`/`topLogprobs` and `TextContent.logprobs` let callers request and receive per-token logprobs from any provider routed through the shared `openai-completions` adapter (OpenAI, GLM/Z.AI, Kimi via Moonshot, Qwen, DeepSeek, and other OpenAI-compatible providers).

Both were added in this fork's own commit history and are not present upstream.

## How this differs from upstream day-to-day

For a normal user, this is a drop-in replacement for upstream: same CLI, same config, same everything else, plus the two extension-API/provider capabilities above. There is no other intentional behavioral divergence — the rest of this fork's own commits (on top of periodic upstream merges) are CI and repo-infrastructure changes (`.github/workflows/*`), not runtime changes.

## Install

This fork isn't published to npm under a different package name, so `npm install @earendil-works/pi-coding-agent` gets you upstream, not this fork. To get this fork:

**Prebuilt binary**: download a platform binary (`pi-<platform>.tar.gz`/`.zip`) from [Astera-org/pi releases](https://github.com/Astera-org/pi/releases). No Node.js required.

**Build from source** (this is a monorepo; the coding agent package depends on several sibling packages):

```bash
git clone https://github.com/Astera-org/pi
cd pi
npm install --ignore-scripts
npm run build
```

This builds `chord`, `tui`, `telemetry`, `ai`, `durable`, `agent`, `sqlite-node`, `protocol`, `client`, `server`, then `coding-agent`, producing the CLI entry point at `packages/coding-agent/dist/bundle/cli.js`. Run it directly with `node packages/coding-agent/dist/bundle/cli.js`, or `npm link` from `packages/coding-agent` to put `pi` on your `PATH`.

## Staying in sync

This fork periodically merges upstream `main` with a real `git merge` (never a rebase or squash) to keep the sync tractable; expect periodic syncs rather than a one-time fork.
